import os
import wandb
import hydra
import torch
import torch.distributed as dist
from torch import autocast
from omegaconf import DictConfig, OmegaConf, open_dict
from collections import defaultdict
import time

from mtp.data.dataloader import DistributedDataLoader
from mtp.utils.distributed import setup_distributed, wrap_model_distributed
from mtp.utils.checkpoint import Checkpoint
from mtp.utils.logger import Logger


def set_deterministic(seed):
    # https://discuss.pytorch.org/t/reproducibility-with-multiple-gpus-not-working/209583
    torch.manual_seed(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    # For this we also need env var CUBLAS_WORKSPACE_CONFIG=:4096:8
    torch.backends.cudnn.deterministic = True


def create_optimizers(raw_model, cfg):
    """Initialize optimizers and schedulers."""
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(.9, .95),
        eps=1e-8,
        fused=True,
        weight_decay=0.1,
    )

    if cfg.training.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_iterations,
            eta_min=cfg.training.learning_rate / 10
        )
    else:
        scheduler = None

    return optimizer, scheduler


@torch.no_grad()
def validation_step(model, val_loader, val_steps, ctx):
    """Run validation."""
    model.eval()
    val_loader.reset()
    val_loss, metrics = 0., defaultdict(lambda: torch.tensor([0.], device=model.device))
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            results = model(x_val, y_val)
            val_loss += results.pop('loss').detach()
            for k, v in results.items():
                if '_loss_' in k:
                    metrics[k] += v.detach()
            del results

    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    val_loss /= val_steps

    for k, v in list(metrics.items()):
        dist.all_reduce(metrics[k], op=dist.ReduceOp.AVG)
        metrics[k] /= val_steps
    return val_loss, metrics


def training_step(model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx):
    """Run single training step."""
    model.train()
    train_loss, metrics = 0., defaultdict(lambda: torch.tensor([0.], device=model.device))
    for i in range(1, train_accumulation_steps + 1):
        x, y = train_loader.next_batch()

        with ctx:
            results = model(x, y)
            loss = results.pop('loss')
            train_loss += loss.detach()
            for k, v in results.items():
                if '_loss_' in k:
                    metrics[k] += v.detach()
            del results

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
    if scheduler is not None:
        scheduler.step()

    model.zero_grad(set_to_none=True)

    dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
    train_loss /= train_accumulation_steps

    for k, v in list(metrics.items()):
        dist.all_reduce(metrics[k], op=dist.ReduceOp.AVG)
        metrics[k] /= train_accumulation_steps

    return train_loss, metrics


def name_exp(cfg):
    if getattr(cfg.training, "expname", None) is not None:
        return cfg.training.expname
    name = cfg.model.name
    if name == 'mtp':
        name = '%s-n=%d-r=%d' % (name, cfg.model.n_token, cfg.model.n_component)
    return name


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="config")
def main(cfg: DictConfig):
    try:

        # NOTE: Below seems needed if freeze=false for some LLMs
        # torch._dynamo.config.optimize_ddp = False

        set_deterministic(cfg.training.random_seed)
        # Set DEVICE env variable, which is used by mtp.utils.distributed
        os.environ['DEVICE'] = cfg.device

        # Initialize distributed setup
        rank, local_rank, world_size, _ = setup_distributed()
        master_process = (rank == 0)

        # Setup logging
        logger = Logger(master_process)

        # ===================== BEGIN MODEL SETUP ============================
        # Initialize model. Use model for checkpoints
        # as this is current pytorch recommendation for saving compiled models
        # https://pytorch.org/get-started/pytorch-2.0/#serialization
        model = hydra.utils.instantiate(cfg.model).model
        optimized_model = wrap_model_distributed(model, local_rank, cfg.compile)

        # Initialize optimizers and schedulers
        logger("Setting up/compiling model...")
        optimizer, scheduler = create_optimizers(optimized_model, cfg)

        # Initialize training context
        ctx = autocast(device_type=cfg.device, dtype=torch.bfloat16)

        # ===================== BEGIN CHECKPOINT SETUP =======================
        if cfg.from_checkpoint is None:
            global_step = 0
            if master_process:
                expname = name_exp(cfg)
                # Setup Wandb
                run = wandb.init(entity="circuit-mtp",
                                 project='mtp',
                                 name=expname,
                                 # entity=os.environ['USER'],
                                 tags=[cfg.data.name, os.environ['USER']],
                                 config=OmegaConf.to_container(cfg))
                wandb.define_metric("*", step_metric="global_step")
                with open_dict(cfg):
                    cfg.expname = expname
                    cfg.wandb_run_id = run.id

                # Hydra sets cwd to the generated folder
                ckp = Checkpoint(folder=os.getcwd(), config=cfg)
                logger(f"Saving config and checkpoints to {ckp.folder}...")
                logger(f"Save model: %s..." % cfg.training.save_model)
                logger(f"Save optimizer: %s..." % cfg.training.save_optimizer)
                ckp.save()
        else:
            # Load the checkpoint to restore training from
            logger(f"Restoring checkpoint {cfg.from_checkpoint}...")
            ckp = Checkpoint.load(cfg.from_checkpoint)
            global_step = ckp.global_step
            cfg = ckp.config

            # Restore the model, optimizer and scheduler from checkpoint
            ckp.restore(model=model, optimizer=optimizer, scheduler=scheduler)

            if master_process:
                # Setup Wandb to resume run by passing wandb id
                run = wandb.init(entity="circuit-mtp",
                                 project='mtp',
                                 name=cfg.expname,
                                 # entity=os.environ['USER'],
                                 tags=[cfg.data.name, os.environ['USER']],
                                 config=OmegaConf.to_container(cfg),
                                 id=cfg.wandb_run_id,
                                 resume='allow',
                                 # Can use resume_from or fork_from if we get access
                                 # resume_from=f"{cfg.wandb_run_id}?_step={global_step}"
                                 )
                wandb.define_metric("*", step_metric="global_step")

        # ===================== BEGIN DATASET SETUP ==========================
        B, T = cfg.training.device_batch_size, cfg.training.sequence_length
        train_loader = DistributedDataLoader(cfg.data.train_bin, B, T, rank, world_size, cfg.device)
        val_loader = DistributedDataLoader(cfg.data.val_bin, B, T, rank, world_size, cfg.device)

        logger(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        logger(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

        # Calculate steps
        val_steps = cfg.training.val_tokens // (B * T * world_size)
        train_accumulation_steps = cfg.training.batch_size // (B * world_size)

        train_loader.reset()
        if global_step > 0:
            # Skip global_step training examples to resume training
            train_loader.seek(global_step * train_accumulation_steps)

        # ===================== BEGIN TRAINING LOOP ==========================
        for step in range(1 + global_step, cfg.training.num_iterations + 1):
            first_step = (step == 1)
            last_step = (step == cfg.training.num_iterations)

            # Below is needed for reproducibility if we use ops that
            # rely on random state. Need to have same seq of random nums.
            # even if we restore from a checkpoint
            torch.manual_seed(step)

            t0 = time.time()
            if cfg.device == 'cuda':
                torch.cuda.synchronize()

            # Training step
            train_loss, train_metrics = training_step(
                optimized_model, train_loader, train_accumulation_steps, optimizer, scheduler, ctx
            )

            if cfg.device == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Validation
            if first_step or last_step or (step % cfg.training.val_loss_every == 0):
                val_loss, val_metrics = validation_step(optimized_model, val_loader, val_steps, ctx)
                logger(f'step:{step}/{cfg.training.num_iterations} val_loss:{val_loss:.4f}')
                if master_process:
                    wandb.log({
                        'global_step': step,
                        'valid/loss': val_loss,
                        **{('valid/%s' % k): v for k, v in val_metrics.items()},
                    })

            # Logging and model saving
            if master_process:
                if last_step or (step % cfg.training.save_model_every == 0):
                    # TODO: save best / do not overwrite best
                    if cfg.training.save_model:
                        ckp.save(global_step=step,
                                 model=model,
                                 optimizer=optimizer if cfg.training.save_optimizer else None,
                                 scheduler=scheduler if cfg.training.save_optimizer else None)
                    logger(f'step:{step}/{cfg.training.num_iterations} Saving model to %s...' % ckp.modelpath)
                current_lr = optimizer.param_groups[0]['lr']
                logger(f"step:{step}/{cfg.training.num_iterations} train_loss:{train_loss.item():.4f} lr:{current_lr:.10f} time/step:{dt:.2f}s")
                wandb.log({
                    'global_step': step,
                    'train/loss': train_loss,
                    **{('train/%s' % k): v for k, v in train_metrics.items()},
                })
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
