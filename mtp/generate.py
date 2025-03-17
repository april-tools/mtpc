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
from transformers import AutoTokenizer
from torch import autocast

from .train import set_deterministic


BATCH_SIZE = 1


def get_huggingface_model(cfg):
    hf_model = None
    if 'lm' in cfg:
        hf_model = cfg.lm.model.from_huggingface
    return hf_model


def load_vocabs(path):
    with open(path, 'rb') as f:
        vocabs = pickle.load(f)
    return dict(encode=lambda x: [vocabs['stoi'][s] for s in x],
                decode=lambda x: ''.join([vocabs['itos'][i] for i in x]))


# TODO: Maybe avoid loading vocabs twice (here and decode)
def encode(text, cfg, device):
    hf_model = get_huggingface_model(cfg)
    if hf_model is not None:
        tokeniser = AutoTokenizer.from_pretrained(hf_model)
        tokeniser.add_bos_token = True
        # TODO: Get reply about what is going on with BOS
        # NOTE: for this model we need a prompt
        # BOS is not added by default by the tokenizer
        if hf_model.endswith('ablation-model-fineweb-edu'):
            assert text != '', 'Empty prompt is not supported for %s, use a prompt.' % hf_model
        x = tokeniser.encode(text, return_tensors='pt').to(device)
    else:
        # Below works for char level model only
        # TODO: Make below BOS - unsure what it is for the encoded docs
        if text is None:
            BOS = 1
            x = torch.full(size=(BATCH_SIZE, 1), fill_value=BOS, dtype=torch.int64, device=device)
        else:
            vocabs = load_vocabs(cfg.data.vocabs)
            # TODO: Need tokenizer here for token models
            x = torch.tensor(vocabs['encode'](text), dtype=torch.int, device=device)
            x = x.unsqueeze(0)
    return x


def decode(xx, cfg):
    hf_model = get_huggingface_model(cfg)
    if hf_model is not None:
        tokeniser = AutoTokenizer.from_pretrained(hf_model)
        text = tokeniser.batch_decode(sequences=xx, skip_special_tokens=True)[0]
    else:
        vocabs = load_vocabs(cfg.data.vocabs)
        text = vocabs['decode'](xx.ravel().tolist())
    return text


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
    parser.add_argument('--use-cache', action='store_true',
                        help='Whether to use a kv cache.')
    parser.add_argument('--random-seed', default=13, type=int,
                        help='The random seed to use for sampling.')
    parser.add_argument('--mode', required=True, choices=['stp', 'mtp'],
                        help='Single Token Prediction (stp) is available both for MTP and autoregressive models. '
                        'MTP is available only for MTP models')
    parser.add_argument('overrides', nargs="*")
    args = parser.parse_args()

    set_deterministic(args.random_seed)

    os.environ['DEVICE'] = args.device
    os.environ['MODE'] = 'generate'

    # Initialize training context
    ctx = autocast(device_type=args.device, dtype=torch.bfloat16)

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
    model = torch.compile(model)
    model.eval()

    x = encode(args.prompt or '', cfg, args.device)

    # Init model in case loading takes additional time - do not use this output
    with ctx:
        tokens = model.generate(x, mode=args.mode, use_cache=args.use_cache)['tokens']

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

    past_key_values, past_last_hidden_states = None, None
    with tqdm.tqdm(total=args.num_tokens) as pbar:
        # Keep track of total number of tokens generated
        while (x.shape[1] - init_length) < args.num_tokens:
            if args.speculative:
                with ctx:
                    outputs = model.self_speculative_generate(
                        x,
                        use_cache=args.use_cache,
                        past_key_values=past_key_values,
                        past_last_hidden_states=past_last_hidden_states
                    )
                tokens = outputs['tokens']
                past_key_values = outputs['past_key_values']
                past_last_hidden_states = outputs['past_last_hidden_states']
                # The current self-speculative decoding implementation always
                # returns at least one extra token. So, we subtract 1 to get
                # the number of accepted tokens from the draft/circuit model
                num_accepted_tokens.append(tokens.shape[1] - 1)
            else:
                with ctx:
                    outputs = model.generate(
                        x,
                        mode=args.mode,
                        use_cache=args.use_cache,
                        past_key_values=past_key_values
                    )
                tokens = outputs['tokens']
                past_key_values = outputs['past_key_values']
            x = torch.cat([x, tokens], dim=1)
            pbar.update(tokens.shape[1])

    if args.device == 'cpu':
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    elif args.device == 'cuda':
        end.record(torch.cuda.current_stream(args.device))
        # Synchronize CUDA Kernels before measuring time
        torch.cuda.synchronize(args.device)
        elapsed_time = start.elapsed_time(end) * 1e-3   # CUDA returns ms
    else:
        raise ValueError('Unexpected device %s' % args.device)

    print('Generation:\n\n', decode(x, cfg))

    tps = args.num_tokens / elapsed_time

    stats['model'] = cfg.model.model._target_
    stats['ntoken'] = n_token
    stats['ncomponent'] = n_component
    stats['speculative'] = args.speculative
    stats['use_kv_cache'] = args.use_cache
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
    stats['checkpoint'] = '%s-%s@0' % (ckp.model.name, ckp.lm.name) if args.checkpoint is None else repr(ckp)
    # Below attributes only exist for MTP
    if 'stp' not in stats['model']:
        stats['beta'] = cfg.model.model.beta
        stats['gamma'] = cfg.model.model.gamma
        stats['kl_type'] = cfg.model.model.kl_type
        stats['kl_algorithm'] = cfg.model.model.kl_algorithm
        stats['token_head_type'] = cfg.model.token_head.expander.expander_type
        stats['token_head_num_transformer_layers'] = cfg.model.token_head.encoder.n_layer
        stats['sum_weight_num_transformer_layers'] = cfg.model.sum_weight_head.encoder.n_layer

    result = json.dumps(stats)

    print(result)
