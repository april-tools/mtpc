import os
import json
import torch
import argparse
import torch.distributed as dist
from torch import autocast
from collections import defaultdict

from mtp.data import DistributedDataLoader
from mtp.utils.distributed import setup_distributed, wrap_model_distributed
from mtp.utils.checkpoint import Checkpoint, load_model_with_overrides

from .train import set_deterministic, validation_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="The checkpointed model (.pth file) to validate")
    parser.add_argument("--device", default='cpu',
                        help="The device to use to run validation.")
    parser.add_argument("--random-seed", default=13, type=int,
                        help="The random seed to use.")
    parser.add_argument("--num-examples", default=1024, type=int,
                        help="The number of validation examples to use.")
    parser.add_argument("--device-batch-size", default=None, type=int,
                        help="The device batch size to use.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    set_deterministic(args.random_seed)
    os.environ["DEVICE"] = args.device

    try:
        # Needed for compile
        torch._dynamo.config.suppress_errors = True
        # torch._dynamo.config.optimize_ddp = False

        # Initialize distributed setup
        rank, local_rank, world_size, _ = setup_distributed()
        master_process = (rank == 0)

        # We need below to compute KL with target model
        args.overrides.append("lm.model.encoder_only=false")
        # args.overrides.append("compile=false")
        model, cfg = load_model_with_overrides(args.checkpoint, args.overrides)

        optimized_model = wrap_model_distributed(model, local_rank, cfg.compile)

        # Initialize training context
        ctx = autocast(device_type=cfg.device, dtype=torch.bfloat16)

        # ===================== BEGIN DATASET SETUP ==========================
        B, T = args.device_batch_size, cfg.training.sequence_length
        val_loader = DistributedDataLoader.resolve(cfg.data.val_bin, cfg.lm.model.from_huggingface, B, T, rank, world_size, cfg.device, split='valid')
        assert (args.num_examples % (world_size * B)) == 0
        val_steps = args.num_examples // (world_size * B)

        stats = dict()
        stats["checkpoint"] = f"{cfg.expname}@{cfg.global_step}"
        # Force computation of both CE and full KL
        model.compute_ce = True
        model.compute_kl = True
        # Compute all KL metrics
        # for kl_algo in ["full", "binary_approx"]:
        for kl_algo in ["full"]:
            model.kl_algorithm = kl_algo

            val_loss, val_metrics = validation_step(optimized_model, val_loader, val_steps, args.num_examples, ctx, print_progress=master_process)
            if master_process:
                stats.update(**{k: v.item() for k, v in val_metrics.items()})
        if master_process:
            result = json.dumps(stats)
            print(result)
    finally:
        dist.destroy_process_group()
