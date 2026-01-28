import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import groupby
from mtp.plots.utils import setup_tueplots
from mtp.tables.utils import parse_filename
import matplotlib.font_manager as fm

# Add the font
fm.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/times.ttf')


def get_label(row):
    r = row["ncomponent"]
    circuit = row["circuit"].upper()
    # if circuit == 'FF':
    #     ss = f"FF@{step:<3}"
    #     return f"{ss:<11}"
    # else:
    return f"{circuit:<6} r={r:<4}"


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


def plot_metric(ax, rows, metric):
    colors = dict()
    for i, (model, stats) in enumerate(groupby(rows, lambda x: x["model"])):
        stats = tuple(sorted(stats, key=lambda x: x["adaptor"]))

        means, stds, unique_steps = [], [], []
        for ntoken, iter_stats in groupby(stats, lambda x: x["adaptor"]):
            runs = []
            for run in iter_stats:
                runs.append(run[metric])
            runs = np.array(runs)
            means.append(np.mean(runs))
            stds.append(np.std(runs))
            unique_steps.append(ntoken)

        means, stds = np.array(means), np.array(stds)

        avg_metric = tuple(row[metric] for row in stats)
        steps = tuple(row["adaptor"] for row in stats)

        label = get_label(stats[-1])

        if label not in colors:
            mean_scatter = ax.scatter(unique_steps, means, label=label, s=40, alpha=0.9)
            color = mean_scatter.get_facecolor()
            colors[label] = color
        else:
            mean_scatter = ax.scatter(unique_steps, means, color=colors[label], s=40, alpha=0.9)

        mean_plot = ax.plot(unique_steps, means, "-", color=colors[label], alpha=0.9)
        # ax.fill_between(
        #     unique_steps, means - stds, means + stds, color=colors[label], alpha=0.1
        # )
        #
        # scatter = ax.scatter(
        #     steps, avg_metric, color=colors[label], s=10, alpha=0.3
        # )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Path to throughput.txt file (list of json).")
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

    assert len(args.ntokens) == 1

    rows = []
    for f in args.results:
        arch = parse_filename(f)["model"]
        with open(f, "r") as f:
            for line in f:
                row = json.loads(line)
                row["model"], step = row["checkpoint"].split("@")
                row["step"] = int(step)
                row["circuit"] = get_model_type(row["model"], row["ncomponent"])
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
                    print(row["circuit"])
                    continue
                if args.steps is not None:
                    if row["step"] not in args.steps:
                        if (row["step"] != 0 or row["circuit"] != "ff"):
                            continue
                if row["argmax"] != (args.decoding == "argmax"):
                    continue
                # row["model"] = row["model"].replace("n-8", "").replace("n-16", "")
                row["model"] = get_label(row) + arch
                row["adaptor"] = 0 if row["adaptor"] == "none" else int(row["adaptor"].rsplit("-", 1)[-1])
                rows.append(row)

    rows = tuple(
        sorted(rows, key=lambda x: (x["model"], x["ntoken"], x["circuit"], x["ncomponent"]))
    )

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.8)
    plt.rcParams['font.family'] = 'Times New Roman'

    metric = "avg_accepted_tokens"
    # metric = "tokens_per_second"

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 4.5), nrows=1, ncols=2)

    plot_metric(ax1, rows, "avg_accepted_tokens")
    plot_metric(ax2, rows, "tokens_per_second")

    ax1.set_ylabel("Mean Accepted Tokens")
    ax1.set_xlabel("# LoRA Layers")
    ax1.set_xticks([0, 1, 2, 4])
    ax2.set_ylabel("Throughput (Bytes/s)")
    ax2.set_xlabel("# LoRA Layers")
    ax2.set_xticks([0, 1, 2, 4])
    if args.decoding == "sampling":
        ax2.legend()
        plt.suptitle(f"Speculative Sampling (n={args.ntokens[0]})", fontsize=20, y=0.92)
    else:
        plt.suptitle(f"Greedy Speculative Decoding (n={args.ntokens[0]})", fontsize=20, y=0.92)


    if args.save:
        save_path = os.path.join(
            "outputs", "plots", f"{args.id}.pdf"
        )
        # Save a pdf
        print(f"Saving plot to {save_path} ...")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        # Also save a png
        save_path = os.path.join(
            "outputs", "plots", f"{args.id}.png"
        )
        print(f"Also, saving plot to {save_path} ...")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()
