#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Profile GPU memory usage for MTP decoding by calling mtp.generate directly.

This script constructs mtp.generate commands for different circuit configurations
and runs them as subprocesses while monitoring peak GPU memory usage via nvidia-smi.
"""

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def get_gpu_memory_mb(device_id: int = 0) -> float:
    """Get current GPU memory usage in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Error getting GPU memory: {e}")
        return 0.0


def build_generate_command(
    circuit: str,
    n_token: int,
    n_component: int,
    mode: str,
    adaptor: str,
    num_tokens: int,
    prompt: str,
    use_cache: bool,
    disable_eos: bool,
    device: str = "cuda",
) -> list[str]:
    """Build the mtp.generate command."""
    cmd = [
        "python",
        "-m",
        "mtp.generate",
        "--num-tokens",
        str(num_tokens),
        "--device",
        device,
        "--prompt",
        prompt,
        "--mode",
        "mtp",
        "--task",
        "chat",
    ]

    if use_cache:
        cmd.append("--use-cache")
    if disable_eos:
        cmd.append("--disable-eos")
    if mode == "speculative":
        cmd.append("--speculative")

    # Add config overrides
    overrides = [
        "model=mtp",
        "lm=evabyte",
        "mt_head=linear-evabyte",
        f"adaptor={adaptor}",
        "lm.model.encoder_only=false",
        "data=tulu3-evabyte-packed",
        "data.vocab_size=320",
        "training.batch_size=1",
        "training.device_batch_size=1",
        f"circuit={circuit}",
        f"circuit.n_token={n_token}",
        f"circuit.n_component={n_component}",
        "compile=false",
    ]

    if circuit == "btree":
        overrides.append("circuit.n_repetition=1")

    cmd.extend(overrides)
    return cmd


def profile_command(
    cmd: list[str],
    device_id: int = 0,
    monitor_interval: float = 0.1,
    timeout: int = 600,
) -> dict:
    """Run command and profile memory usage using nvidia-smi."""

    # Get baseline memory
    baseline_mem = get_gpu_memory_mb(device_id)

    # Start memory monitoring in background thread
    peak_memory = [baseline_mem]
    stop_monitoring = threading.Event()

    def monitor():
        while not stop_monitoring.is_set():
            mem = get_gpu_memory_mb(device_id)
            if mem > 0:  # Only update if we got a valid reading
                peak_memory[0] = max(peak_memory[0], mem)
            time.sleep(monitor_interval)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # Run the command
    success = False
    error_msg = None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None if timeout == 0 else timeout,
        )
        success = result.returncode == 0
        if not success:
            error_msg = f"Return code {result.returncode}: {result.stderr[:500]}"
            print(f"Command failed: {error_msg}")
    except subprocess.TimeoutExpired:
        error_msg = "Command timed out"
        print(error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"Error running command: {error_msg}")
    finally:
        stop_monitoring.set()
        monitor_thread.join(timeout=2)

    # Convert to GB
    peak_mem_gb = peak_memory[0] / 1024.0
    baseline_mem_gb = baseline_mem / 1024.0

    return {
        "peak_mem_gb": peak_mem_gb,
        "mem_after_load_gb": baseline_mem_gb,
        "success": success,
        "error": error_msg,
    }


def make_plots(df: pd.DataFrame, output_dir: str, suffix: str) -> list[str]:
    """Create plots from results (same as profile-v2-cli.py)."""
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
    g.set_axis_labels("Rank r (n_component)", "Peak GPU memory (GB)")
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
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID for nvidia-smi monitoring.",
    )
    parser.add_argument(
        "--adaptor",
        default="none",
        choices=["none", "lora-last-1", "lora-last-2", "lora-last-4"],
        help="Adaptor to use when building commands.",
    )
    parser.add_argument(
        "--prompt", default="Who is Albert Einstein?", help="Prompt used to seed decoding."
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
        "--use-cache", action="store_true", help="Enable KV cache during decoding."
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
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each generation command. Use 0 for no timeout.",
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

    os.makedirs(args.output_dir, exist_ok=True)

    # Build run grid
    run_grid: list[RunConfig] = []
    for circuit in args.circuits:
        for n_tok in args.n_tokens:
            for rank in args.ranks:
                for mode in args.modes:
                    if mode not in ("mtp", "speculative"):
                        raise ValueError(f"Unknown mode '{mode}'")
                    run_grid.append(
                        RunConfig(
                            circuit=circuit, n_token=n_tok, n_component=rank, mode=mode
                        )
                    )

    records: list[dict] = []
    for i, run in enumerate(run_grid):
        print(
            f"\n[{i+1}/{len(run_grid)}] Profiling: circuit={run.circuit} n_token={run.n_token} n_component={run.n_component} mode={run.mode}"
        )

        cmd = build_generate_command(
            circuit=run.circuit,
            n_token=run.n_token,
            n_component=run.n_component,
            mode=run.mode,
            adaptor=args.adaptor,
            num_tokens=args.num_tokens,
            prompt=args.prompt,
            use_cache=args.use_cache,
            disable_eos=args.disable_eos,
            device=args.device,
        )

        print(f"Command: {' '.join(cmd)}")

        result = profile_command(cmd, device_id=args.device_id, timeout=args.timeout)

        record = {
            "circuit": run.circuit,
            "n_token": run.n_token,
            "n_component": run.n_component,
            "mode": run.mode,
            "peak_mem_gb": result["peak_mem_gb"],
            "mem_after_load_gb": result["mem_after_load_gb"],
            "num_generated_tokens": args.num_tokens,
            "prompt_length": len(args.prompt.split()),  # Rough estimate
        }
        records.append(record)

        status = "✓" if result["success"] else "✗"
        print(
            f"{status} Peak memory: {result['peak_mem_gb']:.2f} GB (baseline: {result['mem_after_load_gb']:.2f} GB)"
        )

        # Small delay between runs to let GPU settle
        time.sleep(2)

    # Save results
    df = pd.DataFrame(records)
    suffix = "_with_cache" if args.use_cache else "_no_cache"
    json_path = os.path.join(args.output_dir, f"gpu_memory{suffix}.json")
    csv_path = os.path.join(args.output_dir, f"gpu_memory{suffix}.csv")

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    df.to_csv(csv_path, index=False)

    # Make plots
    plot_paths = make_plots(df, args.output_dir, suffix)

    print(f"\nSaved measurements to {json_path} and {csv_path}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
