import os
import json
import tqdm
import torch
import pickle
import socket
import argparse
import datetime
import numpy as np

from transformers import AutoTokenizer
from itertools import chain
from torch import autocast
from langdetect import detect

from mtp.utils.timestamp import unique_timestamp
from mtp.utils.checkpoint import load_model_with_overrides
from mtp.utils.profile import time_block
from mtp.data import DistributedDataLoader

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


def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception:
        return False  # Handle detection errors


def is_ascii_only(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


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
    warmup: bool = False,
    stop_on_eos=True,
):
    # Init model in case loading takes additional time - do not use this output
    if warmup:
        with ctx:
            _ = model.generate(x, mode=args.mode, use_cache=False)

    assert x.shape[0] == 1
    init_length = x.shape[1]
    generated_tokens, num_generated_tokens, num_accepted_tokens, time_per_call = (
        [],
        [],
        [],
        [],
    )
    past_key_values, head_past_key_values = None, None
    verifier_past_key_values = None
    past_num_tokens = None
    last_hidden_state = None
    acc_tokens = None
    prefill_time = 0

    with tqdm.tqdm(total=args.num_tokens, disable=disable_progress_bar) as pbar, ctx:
        # Keep track of total number of tokens generated
        while (x.shape[1] - init_length) < args.num_tokens:
            with time_block(args.device) as t:
                if args.speculative:
                    if args.argmax:
                        outputs = model.self_speculative_generate_argmax(
                            x,
                            use_cache=args.use_cache,
                            draft_past_key_values=past_key_values,
                            verifier_past_key_values=verifier_past_key_values,
                            head_past_key_values=head_past_key_values,
                            past_num_tokens=past_num_tokens,
                            last_hidden_state=last_hidden_state,
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
                    tokens = outputs["tokens"]
                    acc_tokens = outputs["num_accepted_tokens"]
                    past_key_values = outputs["draft_past_key_values"]
                    verifier_past_key_values = outputs["verifier_past_key_values"]
                    head_past_key_values = outputs["head_past_key_values"]
                    past_num_tokens = outputs["past_num_tokens"]
                    last_hidden_state = outputs["last_hidden_state"]
                elif args.mode == "mtp":
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
                        mode="stp",
                        use_cache=args.use_cache,
                        past_key_values=past_key_values,
                    )
                    tokens = outputs["tokens"]
                    past_key_values = outputs["past_key_values"]

                # Handle prefill time
                if outputs["prefill_time"] != 0:
                    assert (
                        prefill_time == 0
                    ), "Prefill unexpectedly non-zero for more than one forward pass"
                    prefill_time = outputs["prefill_time"]

                # Stop if we generate the EOS token
                x = torch.cat([x, tokens], dim=1)
                generated_tokens.append(tokens)
                num_generated_tokens.append(tokens.shape[1])
                num_accepted_tokens.append(acc_tokens)

            time_per_call.append(t.elapsed_time)
            pbar.update(tokens.shape[1])
            if tokeniser is not None and stop_on_eos:
                if torch.any(tokens == tokeniser.eos_token_id):
                    break
            if print_generation:
                print(decode(tokens), end="")

    generated_tokens = [decode(t) for t in generated_tokens]

    result = {
        "time_per_call": time_per_call,
        "generated_tokens": generated_tokens,
        "num_generated_tokens": num_generated_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "prefill_time": prefill_time,
    }
    return result


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
        default=1.0,
        type=float,
        help="The cumulative probability threshold above which to truncate "
        "the circuit categoricals and sum weights. "
        "1. has no effect while 0. is equivalent to approximate argmax.",
    )
    parser.add_argument(
        "--target-top-p",
        default=1.0,
        type=float,
        help="The cumulative probability threshold above which to truncate "
        "the target model's categorical distribution for next token prediction. "
        "1. has no effect while 0. is equivalent to approximate argmax.",
    )
    parser.add_argument(
        "--dequantize",
        default=False,
        action="store_true",
        help="Whether to dequantize the model before measuring the throughput",
    )
    parser.add_argument(
        "--argmax",
        default=False,
        action="store_true",
        help="Whether to use argmax to get samples from the circuit or the STP model",
    )
    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Whether to compile the model",
    )
    parser.add_argument(
        "--no-stop-on-eos",
        default=False,
        action="store_true",
        help="Do not stop when EOS is generated. We use this for measuring throughput "
        "of models that are noisy (e.g. not trained)",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    assert "MTP_ROOT" in os.environ

    exp_start = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

    set_deterministic(args.random_seed)

    os.environ["DEVICE"] = args.device
    os.environ["MODE"] = "generate"

    assert 0.0 <= args.draft_top_p <= 1.0
    assert 0.0 <= args.target_top_p <= 1.0

    # Initialize training context
    ctx = autocast(device_type=args.device, dtype=torch.bfloat16)

    if args.speculative:
        args.overrides.append("lm.model.encoder_only=false")
    # If args.checkpoint=None, load random initialised model with overrides
    model, cfg = load_model_with_overrides(args.checkpoint, args.overrides)

    model.to(args.device)
    model.eval()

    if args.dequantize:
        model.lm.dequantize()

    if model.lm.has_adapter:
        model.lm.enable_dual_model_inference()

    if args.compile:
        # Enable verbose logging
        model = torch.compile(model)

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
            if args.task == "chat":
                tokeniser = AutoTokenizer.from_pretrained(
                    "EvaByte/EvaByte-SFT", **kwargs
                )
            else:
                tokeniser = AutoTokenizer.from_pretrained("EvaByte/EvaByte", **kwargs)
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
                None,  # Do not batch
                8192,
                0,
                1,
                device="cuda",
                split=split,
                as_iterable=False,
                shuffle=False,
            )
            ds = iter(dl.dataset)

            total_prompts, non_user, diff_lang, non_ascii = 0, 0, 0, 0
            for example in ds:
                total_prompts += 1
                prompt = example["messages"][0]
                # We only add the first turn
                # We also ignore prompts that start with a system prompt (rare)
                if prompt["role"] != "user":
                    non_user += 1
                    continue
                if not is_english(prompt["content"]):
                    diff_lang += 1
                    continue
                if not is_ascii_only(prompt["content"]):
                    non_ascii += 1
                    continue
                prompts.append(
                    {
                        "text": prompt["content"],
                        "id": example["id"],
                        "source": example["source"],
                    }
                )
            print("Loaded %d prompts" % total_prompts)
            print("Filtered out %d prompts where first prompt was not user" % non_user)
            print("Filtered out %d prompts that were non-English" % diff_lang)
            print("Filtered out %d prompts that were non-ascii" % non_ascii)
            print("We now subsample from the %d remaining prompts" % len(prompts))

            # The above padded dataset contains approx 7k examples
            # NOTE that for 3 random seeds there will be some overlap
            # in the selected prompts, but it is negligible
            # In [10]: items = np.arange(7000)
            # In [11]: aa = np.random.choice(items, 100)
            # In [12]: bb = np.random.choice(items, 100)
            # In [13]: cc = np.random.choice(items, 100)
            # In [14]: np.unique(np.hstack([aa, bb, cc])).shape
            # Out[14]: (296,)   # ideally would be 300
            random_state = np.random.RandomState(args.random_seed)
            idxs = random_state.choice(
                len(prompts), args.subsample_prompts, replace=False
            )
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
            prompts = [{"text": prompts[i]} for i in indices]
        else:
            raise ValueError(f"Unknown source {args.prompt_source}")
    else:
        prompts = [{"text": args.prompt, "source": "cli"}]

    prompt_source = "Terminal" if args.prompt is not None else args.prompt_source

    print(f"Computing throughput using {len(prompts)} prompt(s) from {prompt_source}")
    # Encode the prompts either for completion or chat (depending on args.task)
    xs = []
    for prompt in prompts:
        x = encode(prompt["text"], args.device, args.task)
        xs.append(x)

    # The number of generated token at each LLM generation step
    # e.g., it is a list of ones in the case of a STP model or,
    # in the case of speculative decoding, it is a list of numbers of the form #_of_accepted_tokens + 1
    (
        all_elapsed_times,
        all_generated_tokens,
        all_num_generated_tokens,
        all_num_accepted_tokens,
        prefill_times,
    ) = ([], [], [], [], [])

    for i, x in tqdm.tqdm(enumerate(xs), disable=len(prompts) == 1, total=len(prompts)):
        if args.print:
            print(prompts[i])
        result = generate(
            x,
            disable_progress_bar=True,
            print_generation=args.print,
            draft_top_p=args.draft_top_p,
            target_top_p=args.target_top_p,
            warmup=i == 0,
            stop_on_eos=not args.no_stop_on_eos,
        )
        all_elapsed_times.append(result["time_per_call"])
        all_generated_tokens.append(result["generated_tokens"])
        all_num_generated_tokens.append(result["num_generated_tokens"])
        all_num_accepted_tokens.append(result["num_accepted_tokens"])
        prefill_times.append(result["prefill_time"])
        if args.print:
            print("\n")

    # Compute the TPS as the total number of generated tokens (across all prompts) by the total elapsed time
    collapsed_elapsed_times = list(chain.from_iterable(all_elapsed_times))
    collapsed_num_generated_tokens = list(chain.from_iterable(all_num_generated_tokens))
    collapsed_num_accepted_tokens = list(chain.from_iterable(all_num_accepted_tokens))

    total_elapsed_time = sum(collapsed_elapsed_times)
    total_num_generated_tokens = sum(collapsed_num_generated_tokens)
    total_prefill_time = sum(prefill_times)

    tps = total_num_generated_tokens / (total_elapsed_time - total_prefill_time)
    tps_with_prefill = total_num_generated_tokens / total_elapsed_time
    avg_time_per_call = np.mean(collapsed_elapsed_times)

    n_token = 1
    n_component = 1
    # Override with MTP case
    if hasattr(model, "mt_head"):
        n_token = model.circuit.n_token
        n_component = model.circuit.n_component

    my_uuid = unique_timestamp()

    try:
        folder_path = os.path.join(
            os.environ["MTP_ROOT"], "outputs", "results", "generation_output"
        )
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "%s.jsonl" % my_uuid)
        log_entries = []
        for i in range(len(prompts)):
            log = dict()
            log["generated_tokens"] = all_generated_tokens[i]
            log["num_generated_tokens"] = all_num_generated_tokens[i]
            log["num_accepted_tokens"] = all_num_accepted_tokens[i]
            log["elapsed_time"] = [round(t, 6) for t in all_elapsed_times[i]]
            log["prefill_time"] = round(prefill_times[i], 6)
            log["prompt"] = prompts[i]
            log_entries.append("%s\n" % json.dumps(log))
        with open(file_path, "w") as f:
            f.writelines(log_entries)
    except Exception as e:
        print("Error saving additional info: %s" % e)

    stats = dict()
    stats["uuid"] = my_uuid
    stats["exp_start"] = exp_start
    stats["exp_end"] = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
    stats["exp_host"] = socket.gethostname()
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
    stats["avg_time_per_call"] = avg_time_per_call
    if args.speculative:
        num_token_idxs = n_token + 1
        uniq_accepted_toks, hist_accepted_toks = np.unique(
            collapsed_num_accepted_tokens, return_counts=True
        )
        full_hist_accepted_toks = np.zeros(num_token_idxs, dtype=np.int32)
        full_hist_accepted_toks[uniq_accepted_toks] = hist_accepted_toks
        stats["avg_accepted_tokens"] = np.mean(collapsed_num_accepted_tokens)
        stats["hist_accepted_tokens"] = [
            np.arange(num_token_idxs).tolist(),
            full_hist_accepted_toks.tolist(),
        ]
    stats["device"] = args.device
    stats["batch_size"] = BATCH_SIZE
    stats["num_generated_tokens"] = total_num_generated_tokens
    stats["elapsed_time"] = total_elapsed_time
    stats["elapsed_time_without_prefill"] = total_elapsed_time - total_prefill_time
    stats["tokens_per_second"] = tps
    stats["tokens_per_second_with_prefill"] = tps_with_prefill
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
