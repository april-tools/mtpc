import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import groupby
from mtp.plots.utils import setup_tueplots


def get_label(row):
    r = row["ncomponent"]
    circuit = row["circuit"].upper()
    return f"{circuit:<5} r={r:<4}"


def get_model_type(model, ncomponent):
    if "-ff-" in model:
        return "ff"
    if "-cp-" in model:
        if ncomponent == 1:
            return "ff"
        return "cp"
    if "-hmm-" in model:
        return "hmm"
    if "-btree-" in model:
        return "btree"
    return "unknown"


def order_model(model_name):
    score = 100
    if "stp" in model_name.lower():
        score = 5
    if "ff" in model_name.lower():
        score = 4
    elif "cp" in model_name.lower():
        score = 3
    elif "btree" in model_name.lower():
        score = 2
    elif "hmm" in model_name.lower():
        score = 1
    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Path to throughput.txt file (list of json).")
    parser.add_argument(
        "--type",
        choices=["accepted_tokens", "histogram"],
        default="accepted_tokens",
        type=str,
        help="Path to throughput.txt file (list of json).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        type=str,
        help="Device to plot acceptance rate for.",
    )
    parser.add_argument(
        "--ntokens",
        type=int,
        nargs="+",
        help="The number of tokens for the circuit model",
    )
    parser.add_argument(
        "--ncomponents",
        type=int,
        nargs="+",
        help="The number of components for the circuit model",
    )
    parser.add_argument(
        "--circuits",
        type=str,
        default=None,
        nargs="+",
        help="The type of circuits to include in the analysis",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        required=True,
        choices=("argmax", "sampling"),
        help="What decoding to use",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None, help="What step to filter for"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If specified saves the plot instead of interactive plot.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="",
        help="The id of the experiment that will be appended to the filename",
    )
    parser.add_argument(
        "--filter-experiments",
        nargs="+",
        default=None,
        help="Which experiments to keep",
    )

    args = parser.parse_args()

    rows = []
    with open(args.results, "r") as f:
        rows = []
        for line in f:
            row = json.loads(line)
            row["model"], step = row["checkpoint"].split("@")
            row["step"] = int(step)
            row["circuit"] = get_model_type(row["model"], row["ncomponent"])
            if row["circuit"] == "stp":
                row["avg_accepted_tokens"] = 1.
            if (
                args.filter_experiments is not None
                and row["model"] not in args.filter_experiments
            ):
                continue
            if row["ntoken"] not in args.ntokens:
                continue
            if row["ncomponent"] not in args.ncomponents:
                continue
            if args.circuits is not None and row["circuit"] not in args.circuits:
                continue
            if args.steps is not None:
                if row["step"] not in args.steps and row["circuit"] != "stp":
                    continue
            if row["argmax"] != (args.decoding == "argmax"):
                continue
            rows.append(row)

    rows = tuple(
        sorted(rows, key=lambda x: (x["ntoken"], x["circuit"], x["ncomponent"]))
    )

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.8)

    if args.type == "accepted_tokens":
        fig, ax = plt.subplots(figsize=(10, 6), nrows=1, sharex=True)
        # fig, axes = plt.subplots(figsize=(10, 6), nrows=3)

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x["model"])):
            stats = tuple(sorted(stats, key=lambda x: x["step"]))

            means, stds, unique_steps = [], [], []
            for step, iter_stats in groupby(stats, lambda x: x["step"]):
                runs = []
                for run in iter_stats:
                    runs.append(run["avg_accepted_tokens"])
                runs = np.array(runs)
                means.append(np.mean(runs))
                stds.append(np.std(runs))
                unique_steps.append(step)

            means, stds = np.array(means), np.array(stds)

            avg_accepted_tokens = tuple(row["avg_accepted_tokens"] for row in stats)
            steps = tuple(row["step"] for row in stats)

            label = get_label(stats[-1])

            mean_scatter = ax.scatter(unique_steps, means, label=label, s=40, alpha=0.9)
            color = mean_scatter.get_facecolor()

            mean_plot = ax.plot(unique_steps, means, "-", alpha=0.9)
            ax.fill_between(
                unique_steps, means - stds, means + stds, color=color, alpha=0.1
            )

            scatter = ax.scatter(
                steps, avg_accepted_tokens, color=color, s=10, alpha=0.3
            )

        ax.tick_params(axis="both")
        ax.set_ylabel("Mean Accepted Tokens")
        ax.set_xlabel("# Training steps")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), alignment="left")
    elif args.type == "histogram":
        token_range = [0, 8]
        # token_range = tuple(range(ntoken + 1))
        ntoken = row["ntoken"]
        fig, axes = plt.subplots(figsize=(9, 11), nrows=len(token_range))

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x["model"])):
            stats = tuple(sorted(stats, key=lambda x: x["step"]))
            for a, j in enumerate(token_range):
                counts = tuple(row["hist_accepted_tokens"][1][j] for row in stats)
                steps = tuple(row["step"] for row in stats)
                axes[a].plot(steps, counts, "-o", label=get_label(stats[-1]))

        for a, j in enumerate(token_range):
            axes[a].set_title("# times %d token(s) generated" % (j + 1))

        axes[-1].set_xlabel("# Training steps")
        axes[-1].legend(loc="best")
        fig.supylabel("Number of generated tokens")
        plt.suptitle("Histogram of generated tokens over training")
    else:
        raise ValueError("Unknown type option: %s" % args.type)
    if args.save:
        filename = os.path.basename(args.results).replace(".jsonl", "")
        save_path = os.path.join(
            "outputs", "plots", f"{args.type.replace('_', '-')}-{args.id}.pdf"
        )
        # Save a pdf
        print(f"Saving plot to {save_path} ...")
        plt.savefig(save_path, bbox_inches="tight")
        # Also save a png
        save_path = os.path.join(
            "outputs", "plots", f"{args.type.replace('_', '-')}-{args.id}.png"
        )
        print(f"Also, saving plot to {save_path} ...")
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()

    # Do not confusingly estimate these comparisons using all steps
    if args.steps is not None and len(args.steps) == 1:
        df = pd.DataFrame(rows)
        print(df)
        df["model"] = df.apply(get_label, axis=1)

        agg_dfs = []
        for ntoken in args.ntokens:
            sub_df = df[(df["ntoken"] == ntoken) | (df["circuit"] == "stp")]
            baseline_fields = []
            for field in ["FF "]:
                df_match = sub_df["model"].str.contains(field, regex=False)
                if df_match.any():
                    field_value = sub_df["model"][df_match].iloc[0]
                    baseline_fields.append(field_value)
            if len(baseline_fields) > 0:
                agg = (
                    sub_df
                    .groupby(["model", "circuit", "ncomponent"])
                    .agg(
                        {
                            "avg_accepted_tokens": ["mean", "std"],
                        }
                    )
                )
                agg = agg.sort_values(
                    ["circuit", "ncomponent"],
                    ascending=[False, True],
                    key=lambda x: x.map(order_model) if x.name == "circuit" else x,
                )
                agg = agg.droplevel(["circuit", "ncomponent"])
                # agg = agg.drop(["circuit", "ncomponent"], level=1, axis=1)
                for field_value in baseline_fields:
                    baseline = agg.loc[
                        field_value,
                        ("avg_accepted_tokens", "mean"),
                    ]
                    # Normalize
                    agg[("Acceptance Rate", "increase over FF")] = (
                        agg[("avg_accepted_tokens", "mean")] / baseline
                    )
                agg_dfs.append(agg)
        results = pd.concat(dict(zip(args.ntokens, agg_dfs)), names=["ntoken", "model"])
        print(
            results.to_latex(
                float_format="%.2f", multirow=False, label="tab:avg_accepted_tokens"
            )
        )
