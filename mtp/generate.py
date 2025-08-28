import os
import json
import time
import tqdm
import torch
import hydra
import pickle
import argparse
import numpy as np

from mtp.utils.checkpoint import Checkpoint, load_model_with_overrides
from mtp.data import DistributedDataLoader

from transformers import AutoTokenizer
from torch import autocast

from .train import set_deterministic


BATCH_SIZE = 1


def get_huggingface_model(cfg):
    hf_model = None
    if "lm" in cfg:
        hf_model = cfg.lm.model.from_huggingface
    return hf_model


def load_vocabs(path):
    with open(path, "rb") as f:
        vocabs = pickle.load(f)
    return dict(
        encode=lambda x: [vocabs["stoi"][s] for s in x],
        decode=lambda x: "".join([vocabs["itos"][i] for i in x]),
    )


def encode(text, device, task):
    if hf_model is None:
        assert task == "completion", "chat not supported for non-hf models"
        # Below works for char level model only
        # TODO: Make below BOS - unsure what it is for the encoded docs
        if text is None:
            BOS = 1
            x = torch.full(
                size=(BATCH_SIZE, 1), fill_value=BOS, dtype=torch.int64, device=device
            )
        else:
            assert vocabs is not None
            x = torch.tensor(vocabs["encode"](text), dtype=torch.int, device=device)
            x = x.unsqueeze(0)
    else:
        assert tokeniser is not None
        assert (
            text != ""
        ), f"Empty prompt is not supported for {hf_model}, use a prompt."
        if task == "completion":
            x = tokeniser.encode(text, return_tensors="pt")
        elif task == "chat":
            messages = [{"role": "user", "content": text.strip()}]
            x = tokeniser.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"]
        else:
            raise ValueError(f"Unknown task '{task}'")
        x = x.to(device)
    return x


def decode(xx):
    if hf_model is None:
        assert vocabs is not None
        text = vocabs["decode"](xx.ravel().tolist())
    else:
        assert tokeniser is not None
        text = tokeniser.batch_decode(
            sequences=xx, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    return text


def generate(
    x: torch.Tensor,
    disable_progress_bar: bool = True,
    print_generation: bool = False,
    draft_top_p=1.0,
    target_top_p=1.0,
    warmup: bool = False
):
    # Init model in case loading takes additional time - do not use this output
    if warmup:
        with ctx:
            _ = model.generate(x, mode=args.mode, use_cache=False)

    assert x.shape[0] == 1
    init_length = x.shape[1]
    num_tokens = []
    past_key_values, head_past_key_values = None, None
    verifier_past_key_values = None
    past_num_tokens = None
    last_hidden_state = None

    if args.device == "cpu":
        start_time = time.perf_counter()
    elif args.device == "cuda":
        torch.cuda.synchronize(args.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(args.device))
    else:
        raise ValueError("Unexpected device %s" % args.device)

    with tqdm.tqdm(total=args.num_tokens, disable=disable_progress_bar) as pbar, ctx:
        # Keep track of total number of tokens generated
        while (x.shape[1] - init_length) < args.num_tokens:
            if args.speculative:
                if args.argmax:
                    outputs = model.self_speculative_generate_argmax(
                        x,
                        use_cache=args.use_cache,
                        draft_past_key_values=past_key_values,
                        verifier_past_key_values=verifier_past_key_values,
                        head_past_key_values=head_past_key_values,
                        past_num_tokens=past_num_tokens,
                    )
                else:
                    outputs = model.self_speculative_generate(
                        x,
                        use_cache=args.use_cache,
                        draft_past_key_values=past_key_values,
                        verifier_past_key_values=verifier_past_key_values,
                        head_past_key_values=head_past_key_values,
                        past_num_tokens=past_num_tokens,
                        last_hidden_state=last_hidden_state,
                        draft_top_p=draft_top_p,
                        target_top_p=target_top_p,
                    )
                tokens = outputs['tokens']
                past_key_values = outputs['draft_past_key_values']
                verifier_past_key_values = outputs['verifier_past_key_values']
                head_past_key_values = outputs['head_past_key_values']
                past_num_tokens = outputs['past_num_tokens']
                last_hidden_state = outputs['last_hidden_state']
            elif args.mode == 'mtp':
                outputs = model.generate(
                    x,
                    mode="mtp",
                    use_argmax=args.argmax,
                    use_cache=args.use_cache,
                    past_key_values=past_key_values,
                    head_past_key_values=head_past_key_values,
                    draft_top_p=draft_top_p,
                )
                tokens = outputs["tokens"]
                past_key_values = outputs["past_key_values"]
                head_past_key_values = outputs["head_past_key_values"]
            else:
                assert args.mode == "stp"
                outputs = model.generate(
                    x,
                    mode='stp',
                    use_cache=args.use_cache,
                    past_key_values=past_key_values
                )
                tokens = outputs["tokens"]
                past_key_values = outputs["past_key_values"]
            # Stop if we generate the EOS token
            x = torch.cat([x, tokens], dim=1)
            num_tokens.append(tokens.shape[1])
            pbar.update(tokens.shape[1])
            if tokeniser is not None:
                if torch.any(tokens == tokeniser.eos_token_id):
                    break

    if args.device == "cpu":
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
    elif args.device == "cuda":
        end.record(torch.cuda.current_stream(args.device))
        # Synchronize CUDA Kernels before measuring time
        torch.cuda.synchronize(args.device)
        elapsed_time = start.elapsed_time(end) * 1e-3  # CUDA returns ms
    else:
        raise ValueError("Unexpected device %s" % args.device)

    if print_generation:
        print("\nGeneration:\n", decode(x), "\n\n")

    return elapsed_time, num_tokens


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="The checkpointed model (.pth file) to use for generation or "
        "a .yaml config file if we want to initialise a random model.",
    )
    parser.add_argument(
        "--num-tokens", default=1000, type=int, help="Number of tokens to generate."
    )
    parser.add_argument(
        "--device", default="cpu", help="The device to use for generation."
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        help="Prompt to use for generation. If None use the prompts from spec_bench",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["tulu-train", "tulu-valid", "spec-bench"],
        default="tulu-valid",
        type=str,
        help="If prompt=None, we load prompts from a dataset.",
    )
    parser.add_argument(
        "--subsample-prompts",
        type=int,
        default=20,
        help="How many prompts to subsample from source if --prompt is not given",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Whether to use speculative decoding.",
    )
    parser.add_argument(
        "--print", action="store_true", help="Whether to print the generated texts."
    )
    parser.add_argument(
        "--use-cache",
        default=False,
        action="store_true",
        help="Whether to use a kv cache.",
    )
    parser.add_argument(
        "--random-seed",
        default=13,
        type=int,
        help="The random seed to use for sampling.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["stp", "mtp"],
        help="Single Token Prediction (stp) is available both for MTP and autoregressive models. "
        "MTP is available only for MTP models",
    )
    parser.add_argument(
        "--task",
        default="completion",
        choices=["completion", "chat"],
        help="The generation task - completion just extends the prompt, "
        "chat expects the prompt to be a question and uses the chat template "
        "to get an answer from the model",
    )
    parser.add_argument(
        "--draft-top-p",
        default=1.,
        type=float,
        help="The cumulative probability threshold above which to truncate "
        "the circuit categoricals and sum weights. "
        "1. has no effect while 0. is equivalent to approximate argmax.",
    )
    parser.add_argument(
        "--target-top-p",
        default=1.,
        type=float,
        help="The cumulative probability threshold above which to truncate "
        "the target model's categorical distribution for next token prediction. "
        "1. has no effect while 0. is equivalent to approximate argmax.",
    )
    parser.add_argument(
        "--dequantize",
        default=False,
        action="store_true",
        help="Whether to dequantize the model before measuring the throughput"
    )
    parser.add_argument(
        "--argmax",
        default=False,
        action="store_true",
        help="Whether to use argmax to get samples from the circuit or the STP model"
    )
    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Whether to compile the model",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    set_deterministic(args.random_seed)

    os.environ["DEVICE"] = args.device
    os.environ["MODE"] = "generate"

    assert 0. <= args.draft_top_p <= 1.
    assert 0. <= args.target_top_p <= 1.

    # Initialize training context
    ctx = autocast(device_type=args.device, dtype=torch.bfloat16)

    if args.speculative:
        args.overrides.append('lm.model.encoder_only=false')
    # If args.checkpoint=None, load random initialised model with overrides
    model, cfg = load_model_with_overrides(args.checkpoint, args.overrides)

    model.to(args.device)
    if args.compile:
        model = torch.compile(model)
    model.eval()

    if args.dequantize:
        model.lm.dequantize()

    # Load the tokeniser once, if needed
    # Otherwise, load the vocabulary (shakespeare models)
    hf_model = get_huggingface_model(cfg)
    if hf_model is None:
        vocabs = load_vocabs(cfg.data.vocabs)
        tokeniser = None
    else:
        kwargs = {}
        if "EvaByte" in hf_model:
            kwargs["trust_remote_code"] = True
            # For EvaByte, chat eos id is 11, while for completion eos id is 2.
            # Therefore, load different tokenisers depending on the case
            if args.task == 'chat':
                tokeniser = AutoTokenizer.from_pretrained('EvaByte/EvaByte-SFT', **kwargs)
            else:
                tokeniser = AutoTokenizer.from_pretrained('EvaByte/EvaByte', **kwargs)
        tokeniser = AutoTokenizer.from_pretrained(hf_model, **kwargs)
        vocabs = None

    # Load prompts from prompt_source if specific prompt not given
    if args.prompt is None:
        prompts = []
        if args.prompt_source.startswith("tulu"):
            split = args.prompt_source.split("-")[-1]
            assert split in ("train", "valid")
            # The dataset below is a subset of the packed dataset but in padded format
            # for easy use with EvaByte
            dl = DistributedDataLoader.resolve(
                "agrv/tulu-v3-sft-evabyte-padded-seq-len-8192",
                "EvaByte/EvaByte",
                1,
                8192,
                0,
                1,
                device="cuda",
                split=split,
                as_iterable=False,
                shuffle=False,
            )
            ds = iter(dl.dataset)


            for example in ds:
                prompt = example["messages"][0][0]
                # We only add the first turn
                # We also ignore prompts that start with a system prompt (rare)
                if prompt["role"] == "user":
                    prompts.append(prompt["content"])

            # The above padded dataset contains approx 9k examples
            random_state = np.random.RandomState(args.random_seed)
            idxs = random_state.choice(len(prompts), args.subsample_prompts, replace=False)
            prompts = [prompts[idx] for idx in idxs]
            assert len(prompts) == args.subsample_prompts

        elif args.prompt_source == "spec-bench":
            spec_bench_filepath = os.path.join(
                os.environ["MTP_ROOT"], "data", "spec_bench", "question.jsonl"
            )
            # ds = load_dataset("json", spec_bench_filepath)
            with open(spec_bench_filepath, "r") as f:
                for line in f:
                    row = json.loads(line)
                    # Only append first turn
                    prompts.append(row["turns"][0])
            # Make sure same seed => same prompts on which we compute the throughput
            random_state = np.random.RandomState(args.random_seed)
            indices = random_state.permutation(len(prompts))[: args.subsample_prompts]
            prompts = [prompts[i] for i in indices]
        else:
            raise ValueError(f"Unknown source {args.prompt_source}")
    else:
        prompts = [args.prompt]

    prompt_source = "Terminal" if args.prompt is not None else args.prompt_source

    print(f"Computing throughput using {len(prompts)} prompt(s) from {prompt_source}")
    # Encode the prompts either for completion or chat (depending on args.task)
    xs = []
    for prompt in prompts:
        x = encode(prompt, args.device, args.task)
        xs.append(x)

    # The number of generated token at each LLM generation step
    # e.g., it is a list of ones in the case of a STP model or,
    # in the case of speculative decoding, it is a list of numbers of the form #_of_accepted_tokens + 1
    total_num_tokens = []
    # The elapsed time to go through all the prompts
    total_elapsed_time = 0.0

    for i, x in tqdm.tqdm(enumerate(xs), disable=len(prompts) == 1, total=len(prompts)):
        elapsed_time, num_tokens = generate(
            x,
            disable_progress_bar=len(prompts) > 1,
            print_generation=args.print,
            draft_top_p=args.draft_top_p,
            target_top_p=args.target_top_p,
            warmup=i == 0
        )
        total_elapsed_time += elapsed_time
        total_num_tokens.extend(num_tokens)

    # Compute the TPS as the total number of generated tokens (across all prompts) by the total elapsed time
    tps = sum(total_num_tokens) / total_elapsed_time

    n_token = 1
    n_component = 1
    # Override with MTP case
    if hasattr(model, "mt_head"):
        n_token = model.circuit.n_token
        n_component = model.circuit.n_component

    stats = dict()
    stats["model"] = cfg.model.model._target_
    stats["random_seed"] = args.random_seed
    stats["ntoken"] = n_token
    stats["ncomponent"] = n_component
    stats["task"] = args.task
    stats["subsample_prompts"] = args.subsample_prompts
    stats["prompt_source"] = args.prompt_source
    stats["speculative"] = args.speculative
    stats["use_kv_cache"] = args.use_cache
    stats["dequantize"] = args.dequantize
    stats["argmax"] = args.argmax
    stats["draft_top_p"] = args.draft_top_p
    stats["target_top_p"] = args.target_top_p
    if args.speculative:
        # The number of accepted tokens with speculative decoding at each generation step is the number of generated tokens minus one
        total_num_accepted_tokens = list(map(lambda n: n - 1, total_num_tokens))
        num_token_idxs = n_token + 1
        uniq_accepted_toks, hist_accepted_toks = np.unique(
            total_num_accepted_tokens, return_counts=True
        )
        full_hist_accepted_toks = np.zeros(num_token_idxs, dtype=np.int32)
        full_hist_accepted_toks[uniq_accepted_toks] = hist_accepted_toks
        stats["avg_accepted_tokens"] = np.mean(total_num_accepted_tokens)
        stats["hist_accepted_tokens"] = [
            np.arange(num_token_idxs).tolist(),
            full_hist_accepted_toks.tolist(),
        ]
    stats["device"] = args.device
    stats["batch_size"] = BATCH_SIZE
    stats["elapsed_time"] = total_elapsed_time
    stats["tokens_per_second"] = tps
    if args.checkpoint is None:
        stats["checkpoint"] = f"{cfg.model.name}-{cfg.lm.name}@0"
    else:
        stats["checkpoint"] = f"{cfg.expname}@{cfg.global_step}"
    stats["mode"] = args.mode
    # Below attributes only exist for MTP
    if "stp" not in stats["model"]:
        stats["circuit"] = cfg.circuit.name
        stats["beta"] = cfg.model.model.beta
        stats["gamma"] = cfg.model.model.gamma
        stats["kl_type"] = cfg.model.model.kl_type
        stats["kl_algorithm"] = cfg.model.model.kl_algorithm
        stats["expander_type"] = cfg.mt_head.hyperparameters.expander_type
        stats["expander_n_layer"] = cfg.mt_head.hyperparameters.expander_n_layer
        stats["transformer_n_head"] = cfg.mt_head.hyperparameters.transformer_n_head
        stats["transformer_n_layer"] = cfg.mt_head.hyperparameters.transformer_n_layer

    result = json.dumps(stats)

    print(result)
