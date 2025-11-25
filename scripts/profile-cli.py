#!/usr/bin/env python
"""
Profile GPU memory usage for MTP decoding across circuit types and ranks.

This script instantiates MTP models with configurable circuit shapes,
runs a short decoding loop, and records the peak CUDA memory reserved.
The default grid covers CP and BTree circuits across a few values of
the token block size (n) and rank (r), and measures both plain MTP and
self-speculative decoding (verifier enabled).
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

# Ensure repo root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtp.generate import logits_disable_eos
from mtp.train import set_deterministic
from mtp.utils.checkpoint import load_model_with_overrides
from mtp.models.lora_split_lm import LoRASplitLM


GB = 1024**3


@dataclass
class RunConfig:
    circuit: str
    n_token: int
    n_component: int
    mode: str  # "mtp" or "speculative"


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(v) for v in value.split(",")]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Could not parse int list from '{value}'") from exc


def encode_prompt(tokenizer: AutoTokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt.strip()}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    return encoded.to(device)


def build_overrides(circuit: str, n_token: int, n_component: int, base_overrides: list[str]) -> list[str]:
    overrides = list(base_overrides)
    overrides.extend(
        [
            f"circuit={circuit}",
            f"circuit.n_token={n_token}",
            f"circuit.n_component={n_component}",
            "compile=false",
        ]
    )
    if circuit == "btree":
        overrides.append("circuit.n_repetition=1")
    return overrides


def run_generation(
    model,
    input_ids: torch.Tensor,
    num_tokens: int,
    *,
    speculative: bool,
    use_cache: bool,
    draft_top_p: float,
    target_top_p: float,
    disable_eos: bool,
    tokenizer: AutoTokenizer,
) -> None:
    """Run decoding loop in inference mode to populate KV caches and mtp head."""
    past_key_values = None
    head_past_key_values = None
    verifier_past_key_values = None
    past_num_tokens = None
    last_hidden_state = None
    generated = 0

    logit_processor = None
    if disable_eos:
        logit_processor = lambda logits: logits_disable_eos(logits, tokenizer)

    with torch.inference_mode(), autocast(dtype=torch.bfloat16):
        while generated < num_tokens:
            if speculative:
                outputs = model.self_speculative_generate(
                    input_ids,
                    use_cache=use_cache,
                    draft_past_key_values=past_key_values,
                    verifier_past_key_values=verifier_past_key_values,
                    head_past_key_values=head_past_key_values,
                    past_num_tokens=past_num_tokens,
                    last_hidden_state=last_hidden_state,
                    draft_top_p=draft_top_p,
                    target_top_p=target_top_p,
                    logit_processor=logit_processor,
                    legacy=False,
                )
                tokens = outputs["tokens"]
                past_key_values = outputs["draft_past_key_values"]
                verifier_past_key_values = outputs["verifier_past_key_values"]
                head_past_key_values = outputs["head_past_key_values"]
                past_num_tokens = outputs["past_num_tokens"]
                last_hidden_state = outputs["last_hidden_state"]
            else:
                outputs = model.generate(
                    input_ids,
                    mode="mtp",
                    use_argmax=False,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    head_past_key_values=head_past_key_values,
                    logit_processor=logit_processor,
                )
                tokens = outputs["tokens"]
                past_key_values = outputs["past_key_values"]
                head_past_key_values = outputs["head_past_key_values"]

            input_ids = torch.cat([input_ids, tokens], dim=1)
            generated += tokens.shape[1]


def peak_memory_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved() / GB


def profile_run(
    run: RunConfig,
    *,
    device: torch.device,
    prompt: str,
    num_tokens: int,
    use_cache: bool,
    draft_top_p: float,
    target_top_p: float,
    disable_eos: bool,
    base_overrides: list[str],
    tokenizer: AutoTokenizer,
) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    overrides = build_overrides(run.circuit, run.n_token, run.n_component, base_overrides)
    model, cfg = load_model_with_overrides(checkpoint=None, config_overrides=overrides)
    if run.mode == "speculative" and not isinstance(model.lm, LoRASplitLM):
        if getattr(model.lm, "has_adapter", False):
            model.lm.enable_dual_model_inference()
        model.lm = LoRASplitLM.from_lm(model.lm._lm)
    model.to(device)
    model.eval()

    input_ids = encode_prompt(tokenizer, prompt, device)
    prompt_length = input_ids.shape[1]

    base_mem_gb = peak_memory_gb()
    run_generation(
        model,
        input_ids,
        num_tokens,
        speculative=run.mode == "speculative",
        use_cache=use_cache,
        draft_top_p=draft_top_p,
        target_top_p=target_top_p,
        disable_eos=disable_eos,
        tokenizer=tokenizer,
    )
    peak_mem_gb = peak_memory_gb()

    # Clean up before the next run
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "circuit": run.circuit,
        "n_token": run.n_token,
        "n_component": run.n_component,
        "mode": run.mode,
        "peak_mem_gb": peak_mem_gb,
        "mem_after_load_gb": base_mem_gb,
        "num_generated_tokens": num_tokens,
        "prompt_length": prompt_length,
    }


def make_plots(df: pd.DataFrame, output_dir: str, suffix: str) -> list[str]:
    sns.set_theme(style="whitegrid")
    paths: list[str] = []

    df_sorted = df.sort_values(["circuit", "n_token", "n_component"])
    g = sns.relplot(
        data=df_sorted,
        x="n_component",
        y="peak_mem_gb",
        hue="mode",
        style="n_token",
        kind="line",
        col="circuit",
        marker=True,
        facet_kws={"sharey": True, "sharex": True},
    )
    g.set_axis_labels("Rank r (n_component)", "Peak CUDA reserved (GB)")
    g.set_titles("{col_name}")
    g.figure.suptitle("GPU memory vs rank for MTP circuits", y=1.03, fontsize=13)

    plot_path = os.path.join(output_dir, f"gpu_memory_vs_rank{suffix}.png")
    g.figure.savefig(plot_path, bbox_inches="tight", dpi=200)
    paths.append(plot_path)

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device to profile on.")
    parser.add_argument(
        "--adaptor",
        default="none",
        choices=["none", "lora-last-1", "lora-last-2", "lora-last-4"],
        help="Adaptor to use when building overrides.",
    )
    parser.add_argument(
        "--prompt",
        default="Who is Albert Einstein?",
        help="Prompt used to seed decoding.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=256,
        help="Number of tokens to generate during profiling.",
    )
    parser.add_argument(
        "--circuits",
        type=lambda x: x.split(","),
        default="cp,btree",
        help="Comma-separated list of circuits to profile.",
    )
    parser.add_argument(
        "--n-tokens",
        type=parse_int_list,
        default="8,16",
        help="Comma-separated n_token values to profile.",
    )
    parser.add_argument(
        "--ranks",
        type=parse_int_list,
        default="8,16,32",
        help="Comma-separated rank (n_component) values to profile.",
    )
    parser.add_argument(
        "--modes",
        type=lambda x: x.split(","),
        default="mtp,speculative",
        help="Comma-separated decoding modes to profile (mtp,speculative).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable KV cache during decoding.",
    )
    parser.add_argument(
        "--draft-top-p",
        type=float,
        default=1.0,
        help="Top-p for draft sampler (speculative only).",
    )
    parser.add_argument(
        "--target-top-p",
        type=float,
        default=1.0,
        help="Top-p for target sampler (speculative only).",
    )
    parser.add_argument(
        "--disable-eos",
        action="store_true",
        help="Prevent early EOS to keep num_tokens fixed.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "results", "memory_usage"),
        help="Directory to store JSON/CSV and plots.",
    )
    parser.add_argument(
        "--hf-model",
        default="EvaByte/EvaByte-SFT",
        help="HuggingFace identifier for the LM/tokenizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic sampling.",
    )

    args = parser.parse_args()

    if isinstance(args.circuits, str):
        args.circuits = args.circuits.split(",")
    if isinstance(args.n_tokens, str):
        args.n_tokens = parse_int_list(args.n_tokens)
    if isinstance(args.ranks, str):
        args.ranks = parse_int_list(args.ranks)
    if isinstance(args.modes, str):
        args.modes = args.modes.split(",")

    device = torch.device(args.device)
    set_deterministic(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)

    base_overrides = [
        "model=mtp",
        "lm=evabyte",
        "mt_head=linear-evabyte",
        f"adaptor={args.adaptor}",
        "lm.model.encoder_only=false",
        "data=tulu3-evabyte-packed",
        "data.vocab_size=320",
        "training.batch_size=1",
        "training.device_batch_size=1",
    ]

    run_grid: list[RunConfig] = []
    for circuit in args.circuits:
        for n_tok in args.n_tokens:
            for rank in args.ranks:
                for mode in args.modes:
                    if mode not in ("mtp", "speculative"):
                        raise ValueError(f"Unknown mode '{mode}'")
                    run_grid.append(
                        RunConfig(
                            circuit=circuit,
                            n_token=n_tok,
                            n_component=rank,
                            mode=mode,
                        )
                    )

    records: list[dict] = []
    for run in run_grid:
        rec = profile_run(
            run,
            device=device,
            prompt=args.prompt,
            num_tokens=args.num_tokens,
            use_cache=args.use_cache,
            draft_top_p=args.draft_top_p,
            target_top_p=args.target_top_p,
            disable_eos=args.disable_eos,
            base_overrides=base_overrides,
            tokenizer=tokenizer,
        )
        records.append(rec)
        print(
            f"[{run.circuit} n={run.n_token} r={run.n_component} {run.mode}] "
            f"peak={rec['peak_mem_gb']:.2f} GB (load {rec['mem_after_load_gb']:.2f} GB)"
        )

    df = pd.DataFrame(records)
    suffix = "_with_cache" if args.use_cache else "_no_cache"
    json_path = os.path.join(args.output_dir, f"gpu_memory{suffix}.json")
    csv_path = os.path.join(args.output_dir, f"gpu_memory{suffix}.csv")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    df.to_csv(csv_path, index=False)

    plot_paths = make_plots(df, args.output_dir, suffix)

    print(f"\nSaved measurements to {json_path} and {csv_path}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
