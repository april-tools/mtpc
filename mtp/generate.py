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


def encode(text, device):
    if hf_model is None:
        # Below works for char level model only
        # TODO: Make below BOS - unsure what it is for the encoded docs
        if text is None:
            BOS = 1
            x = torch.full(size=(BATCH_SIZE, 1), fill_value=BOS, dtype=torch.int64, device=device)
        else:
            assert vocabs is not None
            # TODO: Need tokenizer here for token models
            x = torch.tensor(vocabs['encode'](text), dtype=torch.int, device=device)
            x = x.unsqueeze(0)
    else:
        assert tokeniser is not None
        # TODO: Get reply about what is going on with BOS
        # NOTE: for this model we need a prompt
        # BOS is not added by default by the tokenizer
        #if hf_model.endswith('ablation-model-fineweb-edu'):
        assert text != '', f'Empty prompt is not supported for {hf_model}, use a prompt.'
        x = tokeniser.encode(text, return_tensors='pt').to(device)
    return x


def decode(xx):
    if hf_model is None:
        assert vocabs is not None
        text = vocabs['decode'](xx.ravel().tolist())
    else:
        assert tokeniser is not None
        text = tokeniser.batch_decode(sequences=xx, skip_special_tokens=True)[0]
    return text


def generate(x: torch.Tensor, disable_progress_bar: bool = True, print_generation=False):
    # Init model in case loading takes additional time - do not use this output
    with ctx:
        _ = model.generate(x, mode=args.mode, use_cache=args.use_cache)['tokens']

    assert x.shape[0] == 1
    init_length = x.shape[1]
    num_tokens = []
    past_key_values, head_past_key_values = None, None

    if args.device == 'cpu':
        start_time = time.perf_counter()
    elif args.device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(args.device))
    else:
        raise ValueError('Unexpected device %s' % args.device)

    with tqdm.tqdm(total=args.num_tokens, disable=disable_progress_bar) as pbar, ctx:
        # Keep track of total number of tokens generated
        while (x.shape[1] - init_length) < args.num_tokens:
            if args.speculative:
                outputs = model.self_speculative_generate(
                    x,
                    use_cache=args.use_cache,
                    past_key_values=past_key_values,
                    head_past_key_values=head_past_key_values
                )
                tokens = outputs['tokens']
                past_key_values = outputs['past_key_values']
                head_past_key_values = outputs['head_past_key_values']
            elif args.mode == 'mtp':
                outputs = model.generate(
                    x,
                    mode='mtp',
                    use_cache=args.use_cache,
                    past_key_values=past_key_values,
                    head_past_key_values=head_past_key_values
                )
                tokens = outputs['tokens']
                past_key_values = outputs['past_key_values']
                head_past_key_values = outputs['head_past_key_values']
            else:
                assert args.mode == 'stp'
                outputs = model.generate(
                    x,
                    use_cache=args.use_cache,
                    past_key_values=past_key_values
                )
                tokens = outputs['tokens']
                past_key_values = outputs['past_key_values']
            x = torch.cat([x, tokens], dim=1)
            num_tokens.append(tokens.shape[1])
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

    if print_generation:
        print('Generation:\n\n', decode(x), '\n')

    return elapsed_time, num_tokens


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
                        help='Prompt to use for generation. If None use the prompts from spec_bench')
    parser.add_argument('--subsample-prompts', type=int, default=0,
                        help='Whether to randomly subsample a number of prompts from spec_bench if --prompt is not given')
    parser.add_argument('--speculative', action='store_true',
                        help='Whether to use speculative decoding.')
    parser.add_argument('--print', action='store_true',
                        help='Whether to print the generated texts.')
    parser.add_argument('--use-cache', default=False, action='store_true',
                        help='Whether to use a kv cache.')
    parser.add_argument('--random-seed', default=13, type=int,
                        help='The random seed to use for sampling.')
    parser.add_argument('--mode', required=True, choices=['stp', 'mtp'],
                        help='Single Token Prediction (stp) is available both for MTP and autoregressive models. '
                        'MTP is available only for MTP models')
    parser.add_argument('--compile', default=False, action='store_true',
                        help="Whether to compile the model")
    parser.add_argument('overrides', nargs="*")
    args = parser.parse_args()

    set_deterministic(args.random_seed)

    os.environ['DEVICE'] = args.device
    os.environ['MODE'] = 'generate'

    # Initialize training context
    ctx = autocast(device_type=args.device, dtype=torch.bfloat16)

    # If we do not pass in a checkpoint, read basic config with overrides
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
        # Hydra tries to make all paths global - urgh.
        cfgfolder = os.path.relpath(f'{ckp.folder}', os.environ['MTP_ROOT'])
        # Use Hydra so that we get the overrides
        with hydra.initialize(version_base=None, config_path=f'../{cfgfolder}', job_name=None):
            cfg = hydra.compose(config_name="config", overrides=args.overrides)
        if args.speculative:
            cfg.lm.model.encoder_only = False
        model = hydra.utils.instantiate(cfg.model).model
        # Restore the checkpoint
        ckp.restore(model=model)
        model.to(args.device)
    if args.compile:
        model = torch.compile(model)
    model.eval()

    # Load the prompts from the 'spec_bench' benchmark, if a particular prompt is not given
    if args.prompt is None:
        prompts = []
        spec_bench_filepath = os.path.join(os.environ['MTP_ROOT'], 'data', 'spec_bench', 'question.jsonl')
        with open(spec_bench_filepath, 'r') as f:
            for line in f:
                row = json.loads(line)
                prompts.extend(row['turns'])
    else:
        prompts = [args.prompt]

    # Load the tokenizer once, if needed
    # Otherwise, load the vocabulary (shakespeare models)
    hf_model = get_huggingface_model(cfg)
    if hf_model is None:
        vocabs = load_vocabs(cfg.data.vocabs)
        tokeniser = None
    else:
        kwargs = {}
        if 'EvaByte' in hf_model:
            kwargs['trust_remote_code'] = True
        tokeniser = AutoTokenizer.from_pretrained(hf_model, **kwargs)
        tokeniser.add_bos_token = True
        vocabs = None

    # Encode all the prompts that can be encoded
    # If not enable to encode (e.g., token missing in the vocabulary), we skip the prompt
    xs = []
    for prompt in prompts:
        try:
            x = encode(prompt, args.device)
        except KeyError:
            continue
        xs.append(x)

    if args.prompt is None and args.subsample_prompts > 0 and len(xs) > args.subsample_prompts:
        # Make sure same seed => same prompts on which we compute the throughput
        random_state = np.random.RandomState(args.random_seed)
        indices = random_state.permutation(len(xs))[:args.subsample_prompts]
        xs = [xs[i] for i in indices]
    print(f"Computing throughput using {len(xs)} prompts")

    # The number of generated token at each LLM generation step
    # e.g., it is a list of ones in the case of a STP model or,
    # in the case of speculative decoding, it is a list of numbers of the form #_of_accepted_tokens + 1
    total_num_tokens = []
    # The elapsed time to go through all the prompts
    total_elapsed_time = 0.0

    for x in tqdm.tqdm(xs, disable=len(prompts) == 1):
        elapsed_time, num_tokens = generate(x, disable_progress_bar=len(prompts) > 1, print_generation=args.print)
        total_elapsed_time += elapsed_time
        total_num_tokens.extend(num_tokens)

    # Compute the TPS as the total number of generated tokens (across all prompts) by the total elapsed time
    tps = sum(total_num_tokens) / total_elapsed_time

    n_token = 1
    n_component = 1
    # Override with MTP case
    if hasattr(model, 'mt_head'):
        n_token = model.circuit.n_token
        n_component = model.circuit.n_component

    stats = dict()
    stats['model'] = cfg.model.model._target_
    stats['ntoken'] = n_token
    stats['ncomponent'] = n_component
    stats['speculative'] = args.speculative
    stats['use_kv_cache'] = args.use_cache
    if args.speculative:
        # The number of accepted tokens with speculative decoding at each generation step is the number of generated tokens minus one
        total_num_accepted_tokens = list(map(lambda n: n - 1, total_num_tokens))
        num_token_idxs = n_token + 1
        uniq_accepted_toks, hist_accepted_toks = np.unique(total_num_accepted_tokens, return_counts=True)
        full_hist_accepted_toks = np.zeros(num_token_idxs, dtype=np.int32)
        full_hist_accepted_toks[uniq_accepted_toks] = hist_accepted_toks
        stats['avg_accepted_tokens'] = np.mean(total_num_accepted_tokens)
        stats['hist_accepted_tokens'] = [np.arange(num_token_idxs).tolist(), full_hist_accepted_toks.tolist()]
    stats['device'] = args.device
    stats['batch_size'] = BATCH_SIZE
    stats['elapsed_time'] = elapsed_time
    stats['tokens_per_second'] = tps
    stats['mode'] = args.mode
    stats['checkpoint'] = '%s-%s@0' % (ckp.model.name, ckp.lm.name) if args.checkpoint is None else repr(ckp)
    # Below attributes only exist for MTP
    if 'stp' not in stats['model']:
        stats['beta'] = cfg.model.beta
        stats['gamma'] = cfg.model.gamma
        stats['kl_type'] = cfg.model.kl_type
        stats['kl_algorithm'] = cfg.model.kl_algorithm
        stats['expander_type'] = cfg.mt_head.hyperparameters.expander_type
        stats['expander_n_layer'] = cfg.mt_head.hyperparameters.expander_n_layer
        stats['transformer_n_head'] = cfg.mt_head.hyperparameters.transformer_n_head
        stats['transformer_n_layer'] = cfg.mt_head.hyperparameters.transformer_n_layer

    result = json.dumps(stats)

    print(result)
