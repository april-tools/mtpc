import os
import json
import tqdm
import hydra
import torch
import torch.distributed as dist
from torch.amp import autocast
from omegaconf import DictConfig
import time

from nanogpt.data.dataloader import DistributedDataLoader
from nanogpt.utils.distributed import setup_distributed, wrap_model_distributed
from nanogpt.utils.logger import Logger


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    # Initialize distributed setup
    # rank, local_rank, world_size, _ = setup_distributed()
    # master_process = (rank == 0)

    myconf = hydra.utils.instantiate(cfg.model)
    model = myconf.model
    
    raw_model = model.to(cfg.device)
    # model = wrap_model_distributed(model, local_rank, cfg.compile)
    # raw_model = model.module

    # Initialize training context
    ctx = autocast(device_type='cuda', dtype=torch.bfloat16)

    NUM_TOKENS = 1000
    # TODO: Make below BOS - unsure what it is for the encoded docs
    BOS = 1
    x = torch.ones(cfg.training.device_batch_size, 1, dtype=torch.int, device=cfg.device)
    x = x * BOS
    # Init model - do not use this output
    tokens = raw_model.generate(x)
    n_token_mtp = getattr(cfg.model, 'n_token', 1)
    assert(tokens.shape[1] == n_token_mtp)

    stats = dict()
    start_time = time.perf_counter()

    with tqdm.tqdm(total=NUM_TOKENS) as pbar:
        # Keep track of total number of tokens generated
        while (x.shape[1] * x.shape[0]) < NUM_TOKENS:
            tokens = raw_model.generate(x)
            x = torch.concat([x, tokens], dim=1)
            pbar.update(tokens.shape[0] * tokens.shape[1])

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    tps = NUM_TOKENS / elapsed_time

    stats['model'] = cfg.model.model._target_
    stats['ntoken'] = n_token_mtp
    stats['ncomponent'] = getattr(cfg.model, 'n_component', 1)
    stats['device'] = cfg.device
    stats['batch_size'] = cfg.training.device_batch_size
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps
    
    result = json.dumps(stats)
    print(result)


if __name__ == "__main__":
    main()
