from deepspeed import DeepSpeedEngine
import deepspeed
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
from mtp.utils.logger import Logger

import types


def deepspeed_zero_stage_config(stage):
    if str(stage) == "0":
        zero_optimization = {"stage": 0}
    elif str(stage) == "1":
        zero_optimization = {"stage": 1, "reduce_bucket_size": 5e8}
    elif str(stage) == "2":
        zero_optimization = {"stage": 2,
                             "allgather_bucket_size": 1e8,
                             "reduce_bucket_size": 1e8,
                             "overlap_comm": True,
                             "reduce_scatter": True,
                             "contiguous_gradients": True}
    elif str(stage) == "3":
        zero_optimization = {"stage": 3,
                             "contiguous_gradients": True,
                             "stage3_max_live_parameters": 1e9,
                             "stage3_max_reuse_distance": 1e9,
                             "stage3_prefetch_bucket_size": 1e7,
                             "stage3_param_persistence_threshold": 1e5,
                             "reduce_bucket_size": 1e7,
                             "sub_group_size": 1e9,
                             "offload_optimizer": {"device": "cpu"},
                             "offload_param": {"device": "cpu"}}
    else:
        raise ValueError(f"{stage}")
    return zero_optimization


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


def create_optimizers(raw_model, cfg, stage):
    """Initialize optimizers and schedulers."""
    if str(stage) == "3":
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            raw_model.parameters(),
            lr=cfg.training.learning_rate,
            betas=(.9, .95),
            eps=1e-8,
            weight_decay=0.1,
        )
    else:
        optimizer = deepspeed.ops.adam.FusedAdam(
            raw_model.parameters(),
            lr=cfg.training.learning_rate,
            betas=(.9, .95),
            eps=1e-8,
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


# @torch.inference_mode()
def deepspeed_validation_step(model_engine, val_loader, val_steps, ctx):
    model_engine.eval()
    val_loader.reset()
    val_loss, metrics = 0., defaultdict(lambda: torch.tensor([0.], device=model_engine.device))
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()

        with ctx:
            results = model_engine(x_val, y_val)
            loss = results.pop("loss")
            val_loss += loss.detach()
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


def deepspeed_training_step(model_engine: DeepSpeedEngine, train_loader, train_accumulation_steps, ctx):
    model_engine.train()
    train_loss, metrics = 0., defaultdict(lambda: torch.tensor([0.], device=model_engine.device))
    for i in range(1, train_accumulation_steps + 1):
        x, y = train_loader.next_batch()

        with ctx:
            results = model_engine(x, y)
            loss = results.pop('loss')
            train_loss += loss.detach()
            for k, v in results.items():
                if '_loss_' in k:
                    metrics[k] += v.detach()
            del results

        model_engine.backward(loss)
        model_engine.step()  # gradient accumulation boundary checking is implemented internally

    # gradient clipping, optimizer.step, and scheduler.step are implemented internally
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
            config_name="deepspeed_config")
def main(cfg: DictConfig):
    try:
        # NOTE: Below seems needed if freeze=false for some LLMs
        # torch._dynamo.config.optimize_ddp = False

        set_deterministic(cfg.training.random_seed)
        # Set DEVICE env variable, which is used by mtp.utils.distributed
        os.environ['DEVICE'] = cfg.device
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        master_process = (rank == 0)

        # Setup logging
        logger = Logger(master_process)

        B, T = cfg.training.device_batch_size, cfg.training.sequence_length

        # Calculate steps
        val_steps = cfg.training.val_tokens // (B * T * world_size)
        train_accumulation_steps = cfg.training.batch_size // (B * world_size)

        # ===================== BEGIN MODEL SETUP ============================
        # Initialize model. Use model for checkpoints
        # as this is current pytorch recommendation for saving compiled models
        # https://pytorch.org/get-started/pytorch-2.0/#serialization
        start_time = time.perf_counter()
        model = hydra.utils.instantiate(cfg.model).model
        total_param = sum([param.nelement() for param in model.parameters()])
        print(f"model size: {total_param / 1e9 if total_param > 1e9 else total_param / 1e6 :.3f}B")
        logger("Setting up model... compile=%r..." % cfg.compile)

        ctx = autocast(device_type=cfg.device, dtype=torch.bfloat16)

        deepspeed_config = {
            "train_micro_batch_size_per_gpu": cfg.training.device_batch_size,
            "gradient_accumulation_steps": train_accumulation_steps,
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
            "zero_optimization": deepspeed_zero_stage_config(cfg.zero_stage),
        }
        # Initialize optimizers and schedulers
        optimizer, scheduler = create_optimizers(model, cfg, cfg.zero_stage)
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            args=types.SimpleNamespace(deepspeed=True, local_rank=rank),
            config=deepspeed_config,
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )

        # ===================== BEGIN CHECKPOINT SETUP =======================
        ckpt_save_dir = os.getcwd()
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
                print(f"Saving config and checkpoints to {ckpt_save_dir}...")
                os.makedirs(ckpt_save_dir, exist_ok=True)
                print(f"Save model: %s..." % cfg.training.save_model)
                print(f"Save optimizer: %s..." % cfg.training.save_optimizer)
        else:
            # # Load the checkpoint to restore training from
            # logger(f"Restoring checkpoint {cfg.from_checkpoint}...")
            # ckp = Checkpoint.load(cfg.from_checkpoint)
            # global_step = ckp.global_step
            # cfg = ckp.config
            #
            # # Restore the model, optimizer and scheduler from checkpoint
            # ckp.restore(model=model, optimizer=optimizer, scheduler=scheduler)

            ckpt_path = model_engine.load_checkpoint(
                load_dir=os.path.dirname(cfg.from_checkpoint),
                tag=os.path.basename(cfg.from_checkpoint),
                load_module_strict=False,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            if ckpt_path is None:
                print(f"rank = {rank} Checkpoint load failed!")
            else:
                print(f"rank = {rank} Checkpoint loaded successfully from {ckpt_path}")

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
            raise NotImplemented

        # ===================== BEGIN DATASET SETUP ==========================
        train_loader = DistributedDataLoader(cfg.data.train_bin, B, T, rank, world_size, cfg.device)
        val_loader = DistributedDataLoader(cfg.data.val_bin, B, T, rank, world_size, cfg.device)

        ntok_train = cfg.training.batch_size * T * cfg.training.num_iterations

        logger(
            f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        logger(
            f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
        logger(f"During training we will see {ntok_train} tokens")
        logger(f"Each validation step will see {cfg.training.val_tokens} tokens")
        if 'shakespeare' not in cfg.data.name:
            assert ntok_train < train_loader.ntok_total, 'Current setup would run multiple epochs on this dataset'

        train_loader.reset()
        if global_step > 0:
            # Skip global_step training examples to resume training
            train_loader.seek(global_step * train_accumulation_steps)

        # Save model at step 0
        # Important: all processes must call this method and not just the process with rank 0.
        # It is because each process needs to save its master weights and scheduler+optimizer states.
        # This method will hang waiting to synchronize with other processes if it’s called just for the process with rank 0.
        if cfg.training.save_model and global_step == 0:
            model_engine.save_checkpoint(save_dir=ckpt_save_dir, tag="0")
            print(f'rank = {rank} step:{global_step}/{cfg.training.num_iterations} -- model saved '
                  f'ckpt_save_dir={ckpt_save_dir}, tag=0')

        # num tokens per step:
        # num tokens = num devices * accumulation steps * device batch size * sequence length
        num_million_tokens_per_step = world_size * train_accumulation_steps * B * T / 1024 ** 2
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
            train_loss, train_metrics = deepspeed_training_step(
                model_engine, train_loader, train_accumulation_steps, ctx
            )

            if cfg.device == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Validation
            if first_step or last_step or (step % cfg.training.val_loss_every == 0):
                val_loss, val_metrics = deepspeed_validation_step(model_engine, val_loader, val_steps, ctx)
                logger(f'step:{step}/{cfg.training.num_iterations} val_loss:{val_loss:.4f}')
                if master_process:
                    wandb.log({
                        'global_step': step,
                        'valid/loss': val_loss,
                        **{('valid/%s' % k): v for k, v in val_metrics.items()},
                    })

            # Logging and model saving
            # Logging
            if master_process:
                current_lr = optimizer.param_groups[0]['lr']
                logger(f"step:{step}/{cfg.training.num_iterations} train_loss:{train_loss.item():.4f} "
                       f"lr:{current_lr:.10f} time/step:{dt:.2f}s "
                       f"M-Tokens/sec:{num_million_tokens_per_step / dt:5f}")
                wandb.log({
                    'global_step': step,
                    'train/loss': train_loss,
                    **{('train/%s' % k): v for k, v in train_metrics.items()},
                })
            # Model saving
            # Important: all processes must call this method and not just the process with rank 0.
            # It is because each process needs to save its master weights and scheduler+optimizer states.
            # This method will hang waiting to synchronize with other processes if it’s called just for the process with rank 0.
            if last_step or (step % cfg.training.save_model_every == 0):
                # TODO: save best / do not overwrite best
                if cfg.training.save_model:
                    model_engine.save_checkpoint(save_dir=os.getcwd(), tag=str(step))
                logger(f'step:{step}/{cfg.training.num_iterations} -- model saved '
                       f'ckpt_save_dir={ckpt_save_dir}, tag={step}')
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
