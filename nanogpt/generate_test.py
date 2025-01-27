import json
import tqdm
import hydra
import torch
import time
from omegaconf import DictConfig


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

    BATCH_SIZE = 1
    NUM_TOKENS = 1024

    # TODO: Make below BOS - unsure what it is for the encoded docs
    BOS = 1
    x = torch.full(size=(BATCH_SIZE, 1), fill_value=BOS, dtype=torch.int64, device=cfg.device)
    n_token_mtp = getattr(cfg.model, 'n_token', 1)
    num_accepted_tokens = []

    stats = dict()
    if cfg.device == 'cpu':
        torch.set_num_threads(64)
        start_time = time.perf_counter()
    elif 'cuda' in cfg.device:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(cfg.device))
    else:
        raise ValueError(f'Unexpected device {cfg.device}')

    with tqdm.tqdm(total=NUM_TOKENS) as pbar:
        if cfg.generate.speculative:
            # Keep track of total number of tokens generated
            # as well as the number of accepted tokens
            while x.shape[1] < NUM_TOKENS:
                tokens = raw_model.self_speculative_generate(x)
                x = torch.cat([x, tokens], dim=1)
                # The current self-speculative decoding implementation always returns
                # at least one extra token. So, we subtract 1 to get the number of accepted
                # tokens from the draft/circuit model
                num_accepted_tokens.append(tokens.shape[1] - 1)
                pbar.update(tokens.shape[1])
        else:
            # Keep track of total number of tokens generated
            while x.shape[1] < NUM_TOKENS:
                tokens = raw_model.generate(x)
                x = torch.cat([x, tokens], dim=1)
                pbar.update(tokens.shape[1])

    if cfg.device == 'cpu':
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    elif 'cuda' in cfg.device:
        end.record(torch.cuda.current_stream(cfg.device))
        torch.cuda.synchronize(cfg.device)  # Synchronize CUDA Kernels before measuring time
        elapsed_time = start.elapsed_time(end) * 1e-3   # CUDA returns ms
    else:
        raise ValueError(f'Unexpected device {cfg.device}')

    tps = NUM_TOKENS / elapsed_time

    stats['model'] = cfg.model.model._target_
    stats['ntoken'] = n_token_mtp
    stats['ncomponent'] = getattr(cfg.model, 'n_component', 1)
    stats['speculative'] = cfg.generate.speculative
    if num_accepted_tokens:
        num_samples = len(num_accepted_tokens)
        stats['avg_accepted_tokens'] = sum(num_accepted_tokens) / num_samples
        stats['prob_all_accepted_tokens'] = len(list(filter(lambda n: n == n_token_mtp, num_accepted_tokens))) / num_samples
    stats['device'] = cfg.device
    stats['batch_size'] = BATCH_SIZE
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps

    result = json.dumps(stats, indent=4)
    print(result)


if __name__ == "__main__":
    main()
