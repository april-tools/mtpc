import os
from typing import Any

import json
import argparse

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from mtp.plots.utils import setup_tueplots


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str, help='Path to throughput.txt file (list of json).')
    parser.add_argument('--ntokens', type=int, nargs='+', help="The number of tokens for the circuit model")
    parser.add_argument('--ncomponents', type=int, nargs='+', help="The number of components for the circuit model")
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', type=str, help='Device to plot throughput for.')
    parser.add_argument('--id', type=str, default="", help="The id of the experiment that will be appended to the filename")

    args = parser.parse_args()

    entries: list[dict[str, Any]] = []
    with open(args.results, 'r') as f:
        for line in f:
            row = json.loads(line)
            if row['device'] != args.device:
                continue
            entry: dict[str, Any] = {}
            if "SingleTokenLM" in row['model']:
                entry['model'] = 'stp'
                entry['circuit'] = None
                entry['ncomponent'] = row['ncomponent']
                entry['ntoken'] = row['ntoken']
                entry['throughput'] = row['tokens_per_second']
                entry['speculative'] = False
                entry['transformer_n_layer'] = None
                entry['hx_accepted_tokens'] = None
                entry['hy_accepted_tokens'] = None
                entry['hy_accepted_tokens_prob'] = None
                entry['avg_accepted_tokens'] = 1.0
                entry['avg_tokens_llm_call'] = 1.0
            elif "MultiTokenLM" in row['model']:
                entry['model'] = row['circuit']
                if row['ntoken'] not in args.ntokens:
                    continue
                if row['ncomponent'] not in args.ncomponents:
                    continue
                if row['speculative']:
                    entry['model'] = f"{entry['model']} (S)"
                else:
                    entry['model'] = f"{entry['model']} (M)"
                entry['ncomponent'] = row['ncomponent']
                entry['ntoken'] = row['ntoken']
                entry['throughput'] = row['tokens_per_second']
                if row['speculative']:
                    entry['hx_accepted_tokens'] = np.array(row['hist_accepted_tokens'][0])
                    entry['hy_accepted_tokens'] = np.array(row['hist_accepted_tokens'][1])
                    entry['hy_accepted_tokens_prob'] = entry['hy_accepted_tokens'] / np.sum(entry['hy_accepted_tokens'])
                    entry['avg_accepted_tokens'] = np.round(row['avg_accepted_tokens'], decimals=2)
                    entry['avg_tokens_llm_call'] = np.round(
                        np.sum(((1 + entry['hx_accepted_tokens']) * entry['hy_accepted_tokens'])) / (2 * np.sum(entry['hy_accepted_tokens'])),
                        decimals=2
                    )
                else:
                    entry['hx_accepted_tokens'] = None
                    entry['hy_accepted_tokens'] = None
                    entry['hy_accepted_tokens_prob'] = None
                    entry['avg_accepted_tokens'] = float(row['ntoken'])
                    entry['avg_tokens_llm_call'] = float(row['ntoken'])
                entry['speculative'] = row['speculative']
                entry['transformer_n_layer'] = row['transformer_n_layer']
            else:
                raise ValueError(f"Unknown model name {row['model']}")
            entries.append(entry)

    def run_identifier(r, show_ntoken: bool = False, show_ncomponent: bool = False) -> str:
        model = r['model']
        res = model
        if model == 'stp':
            return res
        if show_ntoken:
            res += f" n={r['ntoken']}"
        if show_ncomponent:
            res += f" r={r['ncomponent']}"
        transformer_n_layer = r['transformer_n_layer']
        if transformer_n_layer == 0:
            return res
        return f"{res} transf-{int(transformer_n_layer)}"

    df = pd.DataFrame(entries)
    df['run_id'] = df.apply(lambda r: run_identifier(r, show_ncomponent=True), axis=1)

    # Plot throughput by the number of tokens

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.8)

    ax = sb.barplot(
        df,
        x="ntoken",
        y="throughput",
        hue="run_id",
    )
    ax.set_xlabel("# of tokens")
    ax.set_ylabel("throughput (tok/s)")
    ax.grid(linestyle="--", which="major", alpha=0.4, linewidth=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), alignment='left')

    filename = f"throughput-{args.id}.pdf" if args.id else "throughput.pdf"
    plt.savefig(os.path.join("outputs", "plots", filename), bbox_inches='tight')
    plt.clf()
    plt.cla()

    #

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
