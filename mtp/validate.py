import os
import json
import torch
import argparse
import torch.distributed as dist
from torch import autocast
from collections import defaultdict

from mtp.data.dataloader import DistributedDataLoader
from mtp.utils.distributed import setup_distributed, wrap_model_distributed
from mtp.utils.checkpoint import Checkpoint

from .train import set_deterministic, validation_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='The checkpointed model (.pth file) to validate')
    parser.add_argument('--device', default='cpu',
                        help='The device to use to run validation.')
    parser.add_argument('--random-seed', default=13, type=int,
                        help='The random seed to use.')
    parser.add_argument('--device-batch-size', default=None, type=int,
                        help='The device batch size to use.')
    args = parser.parse_args()

    set_deterministic(args.random_seed)
    os.environ['DEVICE'] = args.device

    try:
        # Initialize distributed setup
        rank, local_rank, world_size, _ = setup_distributed()
        master_process = (rank == 0)

        ckp = Checkpoint.load(args.checkpoint)
        cfg = ckp.config
        model = ckp.model
        global_step = ckp.global_step

        if args.device_batch_size is not None:
            cfg.training.device_batch_size = args.device_batch_size

        # Restore the model, optimizer and scheduler from checkpoint
        ckp.restore(model=model, optimizer=None, scheduler=None)

        optimized_model = wrap_model_distributed(model, local_rank, cfg.compile)

        # Initialize training context
        ctx = autocast(device_type=cfg.device, dtype=torch.bfloat16)

        # ===================== BEGIN DATASET SETUP ==========================
        B, T = cfg.training.device_batch_size, cfg.training.sequence_length
        val_loader = DistributedDataLoader(cfg.data.val_bin, B, T, rank, world_size, cfg.device)
        val_steps = cfg.training.val_tokens // (B * T * world_size)

        stats = dict()
        stats['checkpoint'] = '%s@%d' % (cfg.training.expname, global_step)
        # Force computation of both CE and full KL
        model.compute_ce = True
        model.compute_kl = True
        # Compute all KL metrics
        for kl_algo in ['full', 'binary_approx']:
            model.kl_algorithm = kl_algo

            val_loss, val_metrics = validation_step(optimized_model, val_loader, val_steps, ctx)
            if master_process:
                stats.update(**{k: v.item() for k, v in val_metrics.items()})
        if master_process:
            result = json.dumps(stats)
            print(result)
    finally:
        dist.destroy_process_group()
