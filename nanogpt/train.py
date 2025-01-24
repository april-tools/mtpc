import os
import wandb
import hydra
import torch
import torch.distributed as dist
from collections import defaultdict
from torch.amp import autocast
from omegaconf import DictConfig, OmegaConf
import time

from nanogpt.data.dataloader import DistributedDataLoader
from nanogpt.utils.distributed import setup_distributed, wrap_model_distributed
from nanogpt.utils.logger import Logger


def create_optimizers(raw_model, cfg):
    """Initialize optimizers and schedulers."""
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.training.learning_rate,
        fused=True,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_iterations,
        eta_min=cfg.training.learning_rate / 10
    )
    
    return optimizer, scheduler


def validation_step(model, val_loader, val_steps, ctx):
    """Run validation."""
    model.eval()
    val_loader.reset()
    val_loss, val_stp_loss, val_mtp_loss = 0.0, 0.0, 0.0
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            results = model(x_val, y_val, return_logits=False)
            val_loss += results['loss'].detach()
            val_stp_loss += results['stp_loss'].detach()
            if 'mtp_loss' in results:
                val_mtp_loss += results['mtp_loss'].detach()
            del results

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    dist.all_reduce(val_stp_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps
    val_stp_loss /= val_steps

    if val_mtp_loss != 0.0:
        dist.all_reduce(val_mtp_loss, op=dist.ReduceOp.AVG)
        val_mtp_loss /= val_steps
    else:
        val_mtp_loss = None
    return val_loss, val_stp_loss, val_mtp_loss


def training_step(model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx):
    """Run single training step."""
    model.train()
    for i in range(1, train_accumulation_steps+1):
        x, y = train_loader.next_batch()

        with ctx:
            results = model(x, y, return_logits=False)
            loss = results['loss']
            train_loss = loss.detach()
            train_stp_loss = results['stp_loss'].detach()
            if 'mtp_loss' in results:
                train_mtp_loss = results['mtp_loss'].detach()
            else:
                train_mtp_loss = None
        
        if i < train_accumulation_steps and torch.cuda.is_available():
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
    optimizer.step()
    scheduler.step()
        
    model.zero_grad(set_to_none=True)
    
    return train_loss, train_stp_loss, train_mtp_loss


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    try:

        if cfg.ddp:
            # Initialize distributed setup
            rank, local_rank, world_size, _ = setup_distributed()
            master_process = (rank == 0)
        else:
            master_process = True
            rank = 0
            local_rank = 0
            world_size = 1

        if master_process:
            # Setup Wandb
            wandb.init(project='mtp',
                       config=OmegaConf.to_container(cfg))
            wandb.define_metric("*", step_metric="global_step")

            # Below Points to hydra.run.dir (not directly accessible)
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

            # Save current config
            with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
                OmegaConf.save(cfg, f)

            # Setup logging
            logger = Logger(master_process)

        # Initialize data loaders
        B, T = cfg.training.device_batch_size, cfg.training.sequence_length
        train_loader = DistributedDataLoader(cfg.data.train_bin, B, T, rank, world_size, cfg.device)
        val_loader = DistributedDataLoader(cfg.data.val_bin, B, T, rank, world_size, cfg.device)
        
        logger(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        logger(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
        
        # Calculate steps
        val_steps = cfg.training.val_tokens // (B * T * world_size)
        train_accumulation_steps = cfg.training.batch_size // (B * world_size)

        # Initialize model
        myconf = hydra.utils.instantiate(cfg.model)
        model = myconf.model
        
        # If distributed data parallel
        if cfg.ddp:
            model = wrap_model_distributed(model, local_rank, cfg.compile)
            raw_model = model.module
        else:
            raw_model = model
            raw_model = raw_model.to(cfg.device)

        # Initialize optimizers and schedulers
        logger("Setting up/compiling model...")
        optimizer, scheduler = create_optimizers(raw_model, cfg)
        
        # Initialize training context
        ctx = autocast(device_type=cfg.device, dtype=torch.bfloat16)
        
        # Training loop
        train_loader.reset()
        for step in range(1, cfg.training.num_iterations + 1):
            last_step = (step == cfg.training.num_iterations)
            
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Training step
            train_loss, train_stp_loss, train_mtp_loss = training_step(
                model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Validation
            if last_step or (cfg.training.val_loss_every > 0 and step % cfg.training.val_loss_every == 0):
                val_loss, val_stp_loss, val_mtp_loss = validation_step(model, val_loader, val_steps, ctx)
                wandb.log({'valid/loss': val_loss,
                           'valid/stp_loss': val_stp_loss,
                           'valid/mtp_loss': val_mtp_loss,
                           'global_step': step})
                logger(f'step:{step}/{cfg.training.num_iterations} val_loss:{val_loss:.4f}')
            if last_step or (step % cfg.training.save_model_every == 0):
                if master_process:
                    # TODO: save best / do not overwrite best
                    filename = os.path.join(output_dir, 'model@%d.pth' % step)
                    logger(f'step:{step}/{cfg.training.num_iterations} Saving model to %s...' % filename)
                    torch.save(raw_model, filename)

            current_lr = optimizer.param_groups[0]['lr']
            logger(f"step:{step}/{cfg.training.num_iterations} train_loss:{train_loss.item():.4f} lr:{current_lr:.6f} time/step:{dt:.2f}s")
            wandb.log({'train/loss': train_loss,
                       'train/stp_loss': train_stp_loss,
                       'train/mtp_loss': train_mtp_loss,
                       'global_step': step})
    finally:
        if cfg.ddp:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
