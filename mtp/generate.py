import os
import json
import time
import tqdm
import torch
import hydra
import pickle
import argparse
import numpy as np

from mtp.utils.checkpoint import Checkpoint
from .train import set_deterministic


def load_vocabs(path):
    with open(path, 'rb') as f:
        vocabs = pickle.load(f)
    return dict(encode=lambda x: [vocabs['stoi'][s] for s in x],
                decode=lambda x: ''.join([vocabs['itos'][i] for i in x]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='The checkpointed model (.pth file) to use for generation or '
                        'a .yaml config file if we want to initialise a random model.')
    parser.add_argument('--num-tokens', default=1000, type=int,
                        help='Number of tokens to generate.')
    parser.add_argument('--device', default='cpu',
                        help='The device to use for generation.')
    parser.add_argument('--prompt', default=None,
                        help='Prompt to use for generation.')
    parser.add_argument('--speculative', action='store_true',
                        help='Whether to use speculative decoding.')
    parser.add_argument('--random-seed', default=13, type=int,
                        help='The random seed to use for sampling.')
    parser.add_argument('--mode', required=True, choices=['stp', 'mtp'],
                        help='Single Token Prediction (stp) is available both for MTP and autoregressive models. '
                        'MTP is available only for MTP models')
    parser.add_argument('overrides', nargs="*")
    args = parser.parse_args()

    set_deterministic(args.random_seed)

    # TODO: Do we care about changing this?
    BATCH_SIZE = 1
    os.environ['DEVICE'] = args.device

    # If we do not pass in a checkpoint, read config and allow overrides
    if args.checkpoint is None:
        with hydra.initialize(version_base=None, config_path="../configs", job_name=None):
            ckp = hydra.compose(config_name="config", overrides=args.overrides)
        if args.speculative:
            ckp.lm.model.encoder_only = False
        model = hydra.utils.instantiate(ckp.model).model
        model.to(args.device)
        cfg = ckp
    else:
        ckp = Checkpoint.load(args.checkpoint)
        if args.speculative:
            ckp.config.lm.model.encoder_only = False
        model = ckp.model
        cfg = ckp.config

    model.eval()

    vocabs = load_vocabs(cfg.data.vocabs)

    # TODO: Make below BOS - unsure what it is for the encoded docs
    if args.prompt is None:
        BOS = 1
        x = torch.full(size=(BATCH_SIZE, 1), fill_value=BOS, dtype=torch.int64, device=args.device)
    else:
        # TODO: Need tokenizer here for token models
        x = torch.tensor(vocabs['encode'](args.prompt), dtype=torch.int, device=args.device)
        x = x.unsqueeze(0)

    # Init model in case loading takes additional time - do not use this output
    tokens = model.generate(x, mode=args.mode)

    n_token = 1
    n_component = 1
    # Override with MTP case
    if hasattr(model, 'mt_head'):
        n_token = model.mt_head.n_token
        n_component = model.mt_head.n_component
    if args.mode == 'mtp':
        assert tokens.shape[1] == n_token
    else:
        assert tokens.shape[1] == 1

    if args.speculative:
        num_accepted_tokens = []
        assert args.mode == 'mtp', 'Meaningless to use speculative decoding with STP'

    stats = dict()
    if args.device == 'cpu':
        start_time = time.perf_counter()
    elif args.device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(args.device))
    else:
        raise ValueError('Unexpected device %s' % args.device)

    assert x.shape[0] == 1

    init_length = x.shape[1]

    with tqdm.tqdm(total=args.num_tokens) as pbar:
        # Keep track of total number of tokens generated
        while (x.shape[1] - init_length) < args.num_tokens:
            if args.speculative:
                tokens = model.self_speculative_generate(x)
                # The current self-speculative decoding implementation always returns
                # at least one extra token. So, we subtract 1 to get the number of accepted
                # tokens from the draft/circuit model
                num_accepted_tokens.append(tokens.shape[1] - 1)
            else:
                tokens = model.generate(x, mode=args.mode)
            x = torch.cat([x, tokens], dim=1)
            pbar.update(tokens.shape[1])

    if args.device == 'cpu':
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    elif args.device == 'cuda':
        end.record(torch.cuda.current_stream(args.device))
        torch.cuda.synchronize(args.device)  # Synchronize CUDA Kernels before measuring time
        elapsed_time = start.elapsed_time(end) * 1e-3   # CUDA returns ms
    else:
        raise ValueError('Unexpected device %s' % args.device)

    print('Generation:\n\n', vocabs['decode'](x.ravel().tolist()))

    tps = args.num_tokens / elapsed_time

    stats['model'] = cfg.model.model._target_
    stats['ntoken'] = n_token
    stats['ncomponent'] = n_component
    stats['speculative'] = args.speculative
    stats['beta'] = cfg.model.model.beta
    stats['gamma'] = cfg.model.model.gamma
    stats['kl_type'] = cfg.model.model.kl_type
    if args.speculative:
        num_token_idxs = n_token + 1
        uniq_accepted_toks, hist_accepted_toks = np.unique(num_accepted_tokens, return_counts=True)
        full_hist_accepted_toks = np.zeros(num_token_idxs, dtype=np.int32)
        full_hist_accepted_toks[uniq_accepted_toks] = hist_accepted_toks
        stats['avg_accepted_tokens'] = np.mean(num_accepted_tokens)
        stats['hist_accepted_tokens'] = [np.arange(num_token_idxs).tolist(), full_hist_accepted_toks.tolist()]
    stats['device'] = args.device
    stats['batch_size'] = BATCH_SIZE
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps
    stats['mode'] = args.mode
    stats['checkpoint'] = '%s@0' % ckp.name if args.checkpoint is None else repr(ckp)

    result = json.dumps(stats)

    print(result)
