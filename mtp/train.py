import os
import wandb
import hydra
import torch
import torch.distributed as dist
from torch import autocast
from omegaconf import DictConfig, OmegaConf, open_dict
import time

from mtp.data.dataloader import DistributedDataLoader
from mtp.utils.distributed import setup_distributed, wrap_model_distributed
from mtp.utils.checkpoint import Checkpoint
from mtp.utils.logger import Logger


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
            results = model(x_val, y_val, return_stp_loss=True)
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
            results = model(x, y)
            loss = results['loss']
            train_loss = loss.detach()
            if 'mtp_loss' in results:
                train_mtp_loss = results['mtp_loss'].detach()
            else:
                train_mtp_loss = None

        if i < train_accumulation_steps and ctx.device == 'cuda':
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            p.grad /= train_accumulation_steps

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    model.zero_grad(set_to_none=True)

    return train_loss, train_mtp_loss


def name_exp(cfg):
    name = cfg.model.name
    if name == 'mtp':
        name = '%s-n=%d-r=%d' % (name, cfg.model.n_token, cfg.model.n_component)
    return name


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    try:

        # Set DEVICE env variable, which is used by mtp.utils.distributed
        os.environ['DEVICE'] = cfg.device

        # Initialize distributed setup
        rank, local_rank, world_size, _ = setup_distributed()
        master_process = (rank == 0)

        if master_process:
            expname = name_exp(cfg)
            with open_dict(cfg):
                cfg.expname = expname
            # Setup Wandb
            run = wandb.init(project='mtp',
                             name=expname,
                             tags=[cfg.data.name],
                             config=OmegaConf.to_container(cfg))
            wandb.define_metric("*", step_metric="global_step")

            # Below Points to hydra.run.dir (not directly accessible)
            out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

            ckp = Checkpoint(folder=out_dir, config=cfg)
            ckp.save()

            # # Save current config
            # with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            #     OmegaConf.save(cfg, f)

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
        model = hydra.utils.instantiate(cfg.model).model
        model = wrap_model_distributed(model, local_rank, cfg.compile)

        # Initialize optimizers and schedulers
        logger("Setting up/compiling model...")
        optimizer, scheduler = create_optimizers(model, cfg)

        # Initialize training context
        ctx = autocast(device_type=cfg.device, dtype=torch.bfloat16)

        # Training loop
        train_loader.reset()
        for step in range(1, cfg.training.num_iterations + 1):
            last_step = (step == cfg.training.num_iterations)

            t0 = time.time()
            if cfg.device == 'cuda':
                torch.cuda.synchronize()

            # Training step
            train_loss, train_mtp_loss = training_step(
                model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx
            )

            if cfg.device == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Validation
            if last_step or (cfg.training.val_loss_every > 0 and step % cfg.training.val_loss_every == 0):
                val_loss, val_stp_loss, val_mtp_loss = validation_step(model, val_loader, val_steps, ctx)
                logger(f'step:{step}/{cfg.training.num_iterations} val_loss:{val_loss:.4f}')
                if master_process:
                    wandb.log({
                        'valid/loss': val_loss,
                        'valid/stp_loss': val_stp_loss,
                        'valid/mtp_loss': val_mtp_loss,
                        'global_step': step
                    })

            # Logging and model saving
            if master_process:
                if last_step or (step % cfg.training.save_model_every == 0):
                    # TODO: save best / do not overwrite best
                    ckp.save(global_step=step, model=model, optimizer=optimizer, scheduler=scheduler)
                    logger(f'step:{step}/{cfg.training.num_iterations} Saving model to %s...' % ckp.modelpath)
                current_lr = optimizer.param_groups[0]['lr']
                logger(f"step:{step}/{cfg.training.num_iterations} train_loss:{train_loss.item():.4f} lr:{current_lr:.6f} time/step:{dt:.2f}s")
                wandb.log({
                    'train/loss': train_loss,
                    'train/mtp_loss': train_mtp_loss,
                    'global_step': step
                })
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
