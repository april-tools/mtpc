import os
import json
import time
import tqdm
import hydra
import torch
import pickle
import torch.distributed as dist
from torch.amp import autocast
from omegaconf import DictConfig

from nanogpt.data.dataloader import DistributedDataLoader
from nanogpt.utils.distributed import setup_distributed, wrap_model_distributed


def load_vocabs(path):
    with open(path, 'rb') as f:
        vocabs = pickle.load(f)
    return dict(encode=lambda x: [vocabs['stoi'][s] for s in x],
                decode=lambda x: ''.join([vocabs['itos'][i] for i in x]))


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):

    if cfg.checkpoint is None:
        myconf = hydra.utils.instantiate(cfg.model)
        model = myconf.model
    else:
        model = torch.load(cfg.checkpoint)

    raw_model = model.to(cfg.device)

    vocabs = load_vocabs(cfg.data.vocabs)

    # Initialize training context
    ctx = autocast(device_type='cuda', dtype=torch.bfloat16)

    NUM_TOKENS = 1000
    # TODO: Make below BOS - unsure what it is for the encoded docs
    BOS = 1
    x = torch.ones(cfg.training.device_batch_size, 1, dtype=torch.int, device=cfg.device)
    x = x * BOS
    # Init model - do not use this output
    tokens = raw_model.generate(x)
    n_token_mtp = getattr(raw_model.lm_head, 'n_token', 1)
    assert(tokens.shape[1] == n_token_mtp)

    stats = dict()
    if cfg.device == 'cpu':
        start_time = time.perf_counter()
    elif cfg.device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(cfg.device))
    else:
        raise ValueError('Unexpected device %s' % cfg.device)

    with tqdm.tqdm(total=NUM_TOKENS) as pbar:
        # Keep track of total number of tokens generated
        while (x.shape[1] * x.shape[0]) < NUM_TOKENS:
            tokens = raw_model.generate(x)
            x = torch.concat([x, tokens], dim=1)
            pbar.update(tokens.shape[0] * tokens.shape[1])

    if cfg.device == 'cpu':
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    elif cfg.device == 'cuda':
        end.record(torch.cuda.current_stream(cfg.device))
        torch.cuda.synchronize(cfg.device)  # Synchronize CUDA Kernels before measuring time
        elapsed_time = start.elapsed_time(end) * 1e-3   # CUDA returns ms
    else:
        raise ValueError('Unexpected device %s' % cfg.device)

    print('Generation:\n', vocabs['decode'](x.ravel().tolist()))

    tps = NUM_TOKENS / elapsed_time

    stats['model'] = cfg.model.model._target_
    stats['ntoken'] = n_token_mtp
    stats['ncomponent'] = getattr(raw_model.lm_head, 'n_component', 1)
    stats['device'] = cfg.device
    stats['batch_size'] = cfg.training.device_batch_size
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps
    
    result = json.dumps(stats)

    print(result)


if __name__ == "__main__":
    main()
