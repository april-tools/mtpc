import os
import json
import time
import tqdm
import torch
import pickle
import argparse
import numpy as np

from omegaconf import OmegaConf
from hydra.utils import instantiate


def load_vocabs(path):
    with open(path, 'rb') as f:
        vocabs = pickle.load(f)
    return dict(encode=lambda x: [vocabs['stoi'][s] for s in x],
                decode=lambda x: ''.join([vocabs['itos'][i] for i in x]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='The checkpointed model (.pth file) to use for generation or '
                        'a .yaml config file if we want to initialise a random model.')
    parser.add_argument('--num-tokens', default=1000, type=int,
                        help='Number of tokens to generate.')
    parser.add_argument('--device', default='cuda',
                        help='The device to use for generation.')
    parser.add_argument('--prompt', default=None,
                        help='Prompt to use for generation.')
    parser.add_argument('--speculative', action='store_true',
                        help='Whether to use speculative decoding.')
    parser.add_argument('--mode', required=True, choices=['stp', 'mtp'],
                        help='Single Token Prediction (stp) is available both for MTP and autoregressive models. '
                        'MTP is available only for MTP models')
    args = parser.parse_args()

    # TODO: Do we care about changing this?
    BATCH_SIZE = 1
    os.environ['DEVICE'] = args.device

    if args.checkpoint.endswith('.pth'):
        model = torch.load(args.checkpoint,
                           map_location=torch.device(args.device),
                           weights_only=False)

        # Load config used to train the model
        config_folder = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_folder, 'config.yaml')
        cfg = OmegaConf.load(config_path)
        checkpoint = os.path.basename(args.checkpoint)
    elif args.checkpoint.endswith('.yaml'):
        cfg = OmegaConf.load(args.checkpoint)
        model = instantiate(cfg.model).model
        model = model.to(torch.device(args.device))
        checkpoint = 'random'
    else:
        raise ValueError('Invalid checkpoint/config file: %s' % args.checkpoint)
    model.eval()

    vocabs = load_vocabs(cfg.data.vocabs)

    # TODO: Make below BOS - unsure what it is for the encoded docs
    if args.prompt == None:
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
    if args.mode == 'mtp':
        n_token = model.mt_head.n_token
        n_component = model.mt_head.n_component
        assert(tokens.shape[1] == n_token)
    else:
        assert(tokens.shape[1] == 1)

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
    if args.speculative:
        uniq_accepted_toks, hist_accepted_toks = np.unique(num_accepted_tokens, return_counts=True)
        stats['avg_accepted_tokens'] = np.mean(num_accepted_tokens)
        stats['hist_accepted_tokens'] = [uniq_accepted_toks.tolist(), hist_accepted_toks.tolist()]
    stats['device'] = args.device
    stats['batch_size'] = BATCH_SIZE
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps
    stats['mode'] = args.mode
    stats['checkpoint'] = checkpoint

    result = json.dumps(stats)

    print(result)
