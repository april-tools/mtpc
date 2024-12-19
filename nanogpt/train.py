import os
import hydra
import torch
import mlconf
import torch.distributed as dist
from torch.amp import autocast
from omegaconf import DictConfig
import time

from nanogpt.models.gpt import GPT, GPTConfig
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
    val_loss = 0.0
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()
            del loss

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps
    return val_loss


def training_step(model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx):
    """Run single training step."""
    model.train()
    for i in range(1, train_accumulation_steps+1):
        x, y = train_loader.next_batch()

        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        
        if i < train_accumulation_steps:
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
    
    return train_loss


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    # Initialize distributed setup
    rank, local_rank, world_size, _ = setup_distributed()
    master_process = (rank == 0)

    # Setup logging
    logger = Logger(master_process)

    # Initialize data loaders
    B, T = cfg.training.device_batch_size, cfg.training.sequence_length
    train_loader = DistributedDataLoader(cfg.data.train_bin, B, T, rank, world_size)
    val_loader = DistributedDataLoader(cfg.data.val_bin, B, T, rank, world_size)
    
    logger(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    logger(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    
    # Calculate steps
    val_steps = cfg.training.val_tokens // (B * T * world_size)
    train_accumulation_steps = cfg.training.batch_size // (B * world_size)

    myconf = mlconf.Blueprint.from_file(os.path.join(config_path, '/model/example.yaml'))

    # Initialize model
    myconf = myconf.build()
    model = myconf.model
    
    # model = GPT(GPTConfig(**cfg.model))
    model = wrap_model_distributed(model, local_rank, cfg.compile)
    raw_model = model.module

    # Initialize optimizers and schedulers
    optimizer, scheduler = create_optimizers(raw_model, cfg)
    
    # Initialize training context
    ctx = autocast(device_type='cuda', dtype=torch.bfloat16)
    
    # Training loop
    train_loader.reset()
    for step in range(1, cfg.training.num_iterations + 1):
        last_step = (step == cfg.training.num_iterations)
        
        t0 = time.time()
        torch.cuda.synchronize()

        # Training step
        train_loss = training_step(
            model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx
        )

        torch.cuda.synchronize()
        dt = time.time() - t0

        # Validation
        if last_step or (cfg.training.val_loss_every > 0 and step % cfg.training.val_loss_every == 0):
            val_loss = validation_step(model, val_loader, val_steps, ctx)
            logger(f'step:{step}/{cfg.training.num_iterations} val_loss:{val_loss:.4f}')

        current_lr = optimizer.param_groups[0]['lr']
        logger(f"step:{step}/{cfg.training.num_iterations} train_loss:{train_loss.item():.4f} lr:{current_lr:.6f} time/step:{dt:.2f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
