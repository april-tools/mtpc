import os
import tqdm
import hydra
import torch
import mlconf
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
    #
    # # Setup logging
    # logger = Logger(master_process)
    #
    # # Initialize data loaders
    # B, T = cfg.training.device_batch_size, cfg.training.sequence_length
    # train_loader = DistributedDataLoader(cfg.data.train_bin, B, T, rank, world_size)
    # val_loader = DistributedDataLoader(cfg.data.val_bin, B, T, rank, world_size)
    #
    # logger(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    # logger(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    #
    # # Calculate steps
    # val_steps = cfg.training.val_tokens // (B * T * world_size)
    # train_accumulation_steps = cfg.training.batch_size // (B * world_size)

    # The working directory is something like
    # <project-root>/logs/2025-01-06/18-04-52
    # So, compute the config path as follows
    # as to get <project-root>/nanogpt/configs

    # config_path = os.path.join(
    #     os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0],
    #     'nanogpt',
    #     'configs'
    # )
    # myconf = mlconf.Blueprint.from_file(os.path.join(config_path, 'model', cfg.modelconf))

    # Initialize model
    # myconf = myconf.build()
    myconf = hydra.utils.instantiate(cfg.model)
    print(myconf)
    print(myconf.model)
    model = myconf.model
    
    raw_model = model.cuda()
    # model = wrap_model_distributed(model, local_rank, cfg.compile)
    # raw_model = model.module

    # # Initialize optimizers and schedulers
    # optimizer, scheduler = create_optimizers(raw_model, cfg)
    
    # Initialize training context
    ctx = autocast(device_type='cuda', dtype=torch.bfloat16)
    
    # Training loop
    # train_loader.reset()
    # x, y = train_loader.next_batch()
    x = torch.zeros(cfg.training.device_batch_size, 1, dtype=torch.int, device='cuda')
    NUM_TOKENS = 5000

    start_time = time.perf_counter()
    with tqdm.tqdm(total=NUM_TOKENS) as pbar:
        # Keep track of total number of tokens generated
        while (x.shape[1] * x.shape[0]) < NUM_TOKENS:
            tokens = raw_model.generate(x)
            x = torch.concat([x, tokens], dim=1)
            pbar.update(tokens.shape[0] * tokens.shape[1])
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(elapsed_time)


if __name__ == "__main__":
    main()
