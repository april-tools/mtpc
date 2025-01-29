import os
from typing import Any

import json
import argparse

import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from itertools import groupby

from plots.utils import setup_tueplots


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str, help='Path to throughput.txt file (list of json).')
    parser.add_argument('--ncomponent', type=int, default=1, help="The number of components of the circuit model")
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', type=str, help='Device to plot throughput for.')

    args = parser.parse_args()

    entries: list[dict[str, Any]] = []
    with open(args.results, 'r') as f:
        for line in f:
            row = json.loads(line)
            if row['device'] != args.device:
                continue
            entry: dict[str, Any] = {}
            entry['model'] = row['model'].split('.')[-1]
            if entry['model'] == "GPT":
                entry['ncomponent'] = row['ncomponent']
                entry['ntoken'] = row['ntoken']
                entry['throughput'] = row['tokens_per_second']
                entry['hx_accepted_tokens'] = None
                entry['hy_accepted_tokens'] = None
                entry['speculative'] = False
            elif entry['model'] == "MultiTokenLM":
                if row['ncomponent'] != args.ncomponent:
                    continue
                if row['speculative']:
                    entry['model'] = f"{entry['model']} (S)"
                else:
                    entry['model'] = f"{entry['model']} (M)"
                entry['ncomponent'] = row['ncomponent']
                entry['ntoken'] = row['ntoken']
                entry['throughput'] = row['tokens_per_second']
                if row['speculative']:
                    entry['hx_accepted_tokens'] = row['hist_accepted_tokens'][0]
                    entry['hy_accepted_tokens'] = row['hist_accepted_tokens'][1]
                else:
                    entry['hx_accepted_tokens'] = None
                    entry['hy_accepted_tokens'] = None
                entry['speculative'] = row['speculative']
            else:
                raise ValueError(f"Unknown model name {row['model']}")
            entries.append(entry)

    df = pd.DataFrame(entries)


    # Plot throughput by the number of tokens

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.8)

    ax = sb.barplot(
        df,
        x="ntoken",
        y="throughput",
        hue="model"
    )

    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)

    plt.savefig(os.path.join("outputs", "plots", "throughput.pdf"))
    plt.clf()
    plt.cla()

    #

    # Plot number of accepted tokens per multi token model

    df = df[df['speculative'] == True]
    df['model-ntoken'] = df.apply(lambda r: f"{r['model']} (n={r['ntoken']})", axis=1)
    if len(df) == 0:
        exit()
    df = df.explode(['hx_accepted_tokens', 'hy_accepted_tokens'])

    ax = sb.barplot(
        df,
        x="hx_accepted_tokens",
        y="hy_accepted_tokens",
        hue="model-ntoken"
    )
    plt.ylabel("frequency")
    plt.xlabel("accepted tokens")

    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)

    plt.savefig(os.path.join("outputs", "plots", "acceptance-rate-lenient.pdf"))
