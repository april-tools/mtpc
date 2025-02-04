import os
import json
import time
import tqdm
import torch
import pickle
import argparse
from omegaconf import OmegaConf


def load_vocabs(path):
    with open(path, 'rb') as f:
        vocabs = pickle.load(f)
    return dict(encode=lambda x: [vocabs['stoi'][s] for s in x],
                decode=lambda x: ''.join([vocabs['itos'][i] for i in x]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='The checkpointed model (.pth file) to use for generation.')
    parser.add_argument('--num-tokens', default=1000, type=int,
                        help='Number of tokens to generate.')
    parser.add_argument('--device', default='cuda',
                        help='The device to use for generation.')
    parser.add_argument('--prompt', default=None,
                        help='Prompt to use for generation.')
    parser.add_argument('--mode', required=True, choices=['stp', 'mtp'],
                        help='Single Token Prediction (stp) is available both for MTP and autoregressive models. '
                        'MTP is available only for MTP models')
    args = parser.parse_args()

    # TODO: Do we care about changing this?
    BATCH_SIZE = 1

    model = torch.load(args.checkpoint, map_location=torch.device(args.device))
    model.eval()

    # Load config used to train the model
    config_folder = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_folder, 'config.yaml')
    cfg = OmegaConf.load(config_path)

    vocabs = load_vocabs(cfg.data.vocabs)

    # TODO: Make below BOS - unsure what it is for the encoded docs
    if args.prompt == None:
        BOS = 1
        x = torch.ones((BATCH_SIZE, 1), dtype=torch.int, device=args.device)
        x = x * BOS
    else:
        # TODO: Need tokenizer here for token models
        x = torch.tensor(vocabs['encode'](args.prompt), dtype=torch.int, device=args.device)
        x = x.unsqueeze(0)

    # Init model in case loading takes additional time - do not use this output
    tokens = model.generate(x, mode=args.mode)


    if args.mode == 'mtp':
        n_token_mtp = getattr(model.lm_head, 'n_token', 1)
        assert(tokens.shape[1] == n_token_mtp)
    else:
        n_token_mtp = 1
        assert(tokens.shape[1] == 1)

    stats = dict()
    if args.device == 'cpu':
        start_time = time.perf_counter()
    elif args.device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(args.device))
    else:
        raise ValueError('Unexpected device %s' % args.device)

    init_length = x.shape[1] * x.shape[0]

    with tqdm.tqdm(total=args.num_tokens) as pbar:
        # Keep track of total number of tokens generated
        while (x.shape[1] * x.shape[0] - init_length) < args.num_tokens:
            tokens = model.generate(x, mode=args.mode)
            x = torch.concat([x, tokens], dim=1)
            pbar.update(tokens.shape[0] * tokens.shape[1])

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
    stats['ntoken'] = n_token_mtp
    stats['ncomponent'] = getattr(model.lm_head, 'n_component', 1)
    stats['device'] = args.device
    stats['batch_size'] = BATCH_SIZE
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps
    stats['mode'] = args.mode

    result = json.dumps(stats)

    print(result)
