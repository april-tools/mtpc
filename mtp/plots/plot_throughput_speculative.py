import os
from typing import Any

import json
import argparse

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from mtp.plots.utils import setup_tueplots


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


def run_identifier(r, show_ntoken: bool = False, show_ncomponent: bool = False) -> str:
    model = r["model"]
    res = model.upper()
    if model == "stp":
        return res
    if show_ntoken:
        res += f" n={r['ntoken']}"
    if show_ncomponent:
        res += f" r={r['ncomponent']}"
    transformer_n_layer = r["transformer_n_layer"]
    if transformer_n_layer == 0:
        return res
    return f"{res} transf-{int(transformer_n_layer)}"


def read_results(filename):
    entries: list[dict[str, Any]] = []
    with open(filename, "r") as f:
        for line in f:
            row = json.loads(line)
            _, step = row["checkpoint"].split("@")
            row["step"] = int(step)
            if row["device"] != args.device:
                continue
            if row["argmax"] != (args.decoding == "argmax"):
                continue
            if args.step is not None:
                if args.step != row["step"] and "SingleTokenLM" not in row["model"]:
                    continue
            entry: dict[str, Any] = {}
            if "SingleTokenLM" in row["model"]:
                if 1 not in args.ntokens:
                    continue
                entry["model"] = "stp"
                entry["circuit"] = None
                entry["ncomponent"] = row["ncomponent"]
                entry["ntoken"] = row["ntoken"]
                entry["throughput"] = row["tokens_per_second"]
                entry["speculative"] = False
                entry["transformer_n_layer"] = None
                entry["hx_accepted_tokens"] = None
                entry["hy_accepted_tokens"] = None
                entry["hy_accepted_tokens_prob"] = None
                entry["avg_accepted_tokens"] = 1.0
                entry["avg_tokens_llm_call"] = 1.0
            elif "MultiTokenLM" in row["model"]:
                if args.adaptor is not None:
                    adaptor = row.get("adaptor") or None
                    if args.adaptor != adaptor:
                        continue
                entry["model"] = row["circuit"]
                if row["ntoken"] not in args.ntokens:
                    continue
                if row["ncomponent"] not in args.ncomponents:
                    continue
                if row["speculative"]:
                    entry["model"] = f"{entry['model']} (S)"
                else:
                    entry["model"] = f"{entry['model']} (M)"
                entry["model"] = entry["model"].replace("fully_factorized", "ff")
                entry["ncomponent"] = row["ncomponent"]
                entry["ntoken"] = row["ntoken"]
                entry["adaptor"] = row.get("adaptor", None)
                entry["throughput"] = row["tokens_per_second"]
                if row["speculative"]:
                    entry["hx_accepted_tokens"] = np.array(
                        row["hist_accepted_tokens"][0]
                    )
                    entry["hy_accepted_tokens"] = np.array(
                        row["hist_accepted_tokens"][1]
                    )
                    entry["hy_accepted_tokens_prob"] = entry[
                        "hy_accepted_tokens"
                    ] / np.sum(entry["hy_accepted_tokens"])
                    entry["avg_accepted_tokens"] = np.round(
                        row["avg_accepted_tokens"], decimals=2
                    )
                    entry["avg_tokens_llm_call"] = np.round(
                        np.sum(
                            (
                                (1 + entry["hx_accepted_tokens"])
                                * entry["hy_accepted_tokens"]
                            )
                        )
                        / (2 * np.sum(entry["hy_accepted_tokens"])),
                        decimals=2,
                    )
                else:
                    entry["hx_accepted_tokens"] = None
                    entry["hy_accepted_tokens"] = None
                    entry["hy_accepted_tokens_prob"] = None
                    entry["avg_accepted_tokens"] = float(row["ntoken"])
                    entry["avg_tokens_llm_call"] = float(row["ntoken"])
                entry["speculative"] = row["speculative"]
                entry["transformer_n_layer"] = row["transformer_n_layer"]
            else:
                raise ValueError(f"Unknown model name {row['model']}")
            entry["num_generated_tokens"] = row["num_generated_tokens"]
            entries.append(entry)
    return entries


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", type=str, help="Path to throughput.txt file (list of json)."
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
        "--decoding",
        type=str,
        required=True,
        choices=("argmax", "sampling"),
        help="What decoding to use",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="What step to filter for"
    )
    parser.add_argument(
        "--adaptor",
        type=str,
        default=None,
        choices=("none", "lora-last-16"),
        help="Whether to filter for adaptor",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        type=str,
        help="Device to plot throughput for.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="",
        help="The id of the experiment that will be appended to the filename",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If specified saves the plot instead of interactive plot.",
    )

    args = parser.parse_args()

    entries = read_results(args.results)

    df = pd.DataFrame(entries)
    df["run_id"] = df.apply(lambda r: run_identifier(r, show_ncomponent=True), axis=1)
    df = df.sort_values(
        ["model", "ncomponent"],
        ascending=[False, True],
        key=lambda x: x.map(order_model) if x.name == "model" else x,
    )

    # Plot throughput by the number of tokens

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.8)

    ax = sb.barplot(
        df,
        x="ntoken",
        y="throughput",
        hue="run_id",
    )
    # Uncomment below for numbers on bars
    # for bar in ax.patches:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2.,
    #             .2,
    #             f'{height:.2f}',
    #             ha='center', va='bottom',
    #             color='white', weight='bold')  # Make visible against bar color
    ax.set_xlabel("# of tokens")
    ax.set_ylabel("throughput (tok/s)")
    ax.grid(linestyle="--", which="major", alpha=0.4, linewidth=0.6)
    # ax.legend(loc='best')
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), alignment="left")

    if args.save:
        filename = f"throughput-{args.id}.pdf" if args.id else "throughput.pdf"
        save_path = os.path.join("outputs", "plots", filename)
        print(f"Saving plot to {save_path} ...")
        plt.savefig(save_path, bbox_inches="tight")

        filename = f"throughput-{args.id}.png" if args.id else "throughput.png"
        save_path = os.path.join("outputs", "plots", filename)
        print(f"Also, saving plot to {save_path} ...")
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()

    plt.clf()
    plt.cla()

    # Do not confusingly estimate these comparisons using all steps
    if args.step is not None:

        agg_dfs = []
        for ntoken in args.ntokens:
            sub_df = df[(df["ntoken"] == ntoken) | (df["model"] == "stp")]
            baseline_fields = []
            for field in ["FF (M) r=1", "FF (S) r=1", "STP"]:
                df_match = sub_df["run_id"].str.contains(field, regex=False)
                if df_match.any():
                    field_value = sub_df["run_id"][df_match].iloc[0]
                    baseline_fields.append(field_value)
            if len(baseline_fields) > 0:
                agg = sub_df.groupby(["run_id", "model", "ncomponent"]).agg(
                    {
                        "throughput": ["mean", "std"],
                    }
                )
                agg = agg.sort_values(
                    ["model", "ncomponent"],
                    ascending=[False, True],
                    key=lambda x: x.map(order_model) if x.name == "model" else x,
                )
                agg = agg.droplevel(["model", "ncomponent"])
                for field_value in baseline_fields:
                    # Get baseline using the specific index
                    baseline = agg.loc[(field_value), ("throughput", "mean")]
                    # Normalize
                    agg[("throughput", "speed-up over %s" % field_value)] = (
                        agg[("throughput", "mean")] / baseline
                    )
                # Drop STP from the larger ntoken
                if ntoken > 1:
                    agg = agg.drop(index='STP')
                agg_dfs.append(agg)
        results = pd.concat(dict(zip(args.ntokens, agg_dfs)), names=["ntoken", "model"])
        print(
            results.to_latex(
                float_format="%.2f", multirow=False, label="tab:throughput"
            )
        )

    # Plot number of accepted tokens per multi token model
    # unique_ntokens = np.unique([e['ntoken'] for e in entries]).tolist()
    # if unique_ntokens[0] == 1:
    #     del unique_ntokens[0]
    # for ntoken in unique_ntokens:
    #     df = pd.DataFrame(entries)
    #     df = df[df['ntoken'] == ntoken]
    #     df = df[df['speculative'] == True]
    #     df['hy_accepted_tokens_prob'] = df.apply(lambda r: r['hy_accepted_tokens'] / np.sum(r['hy_accepted_tokens']), axis=1)
    #     df['run_id'] = df.apply(
    #         lambda r: f"{run_identifier(r, show_ncomponent=True)} (A-rate={r['avg_accepted_tokens']}, T-rate={r['avg_tokens_llm_call']})",
    #         axis=1
    #     )
    #     if len(df) == 0:
    #         break
    #     df = df.explode(['hx_accepted_tokens', 'hy_accepted_tokens', 'hy_accepted_tokens_prob'])

    #     ax = sb.barplot(
    #         df,
    #         x="hx_accepted_tokens",
    #         y="hy_accepted_tokens_prob",
    #         hue="run_id"
    #     )
    #     ax.set_ylabel("probability")
    #     ax.set_xlabel("accepted tokens")

    #     ax.grid(linestyle="--", which="major", alpha=0.4, linewidth=0.6)
    #     ax.legend(loc='upper left', bbox_to_anchor=(1, 1), alignment='left')

    #     filename = f"acceptance-rate-{args.id}-n-{ntoken}.pdf" if args.id else f"acceptance-rate-n-{ntoken}.pdf"
    #     plt.savefig(os.path.join("outputs", "plots", filename))
    #     plt.clf()
    #     plt.cla()

    # ax = sb.barplot(
    #     df,
    #     x="ntoken",
    #     y="num_generated_tokens",
    #     hue="run_id",
    # )
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1), alignment="left")
    # plt.tight_layout()
    # plt.show()
