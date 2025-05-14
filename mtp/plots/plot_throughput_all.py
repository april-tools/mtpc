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
    parser.add_argument('results', type=str, help='Path to throughput.jsonl file (list of json).')
    parser.add_argument('--use-cache', action='store_true', default=False, help="Whether to show the results with KV cache enabled")
    parser.add_argument('--n_token', type=int, default=8, help="The number of tokens for the circuit model")
    parser.add_argument('--n_components', type=int, nargs='+', help="The number of components for the circuit model")
    parser.add_argument('--id', type=str, default="", help="The id of the experiment that will be appended to the filename")

    args = parser.parse_args()

    entries: list[dict[str, Any]] = []
    with open(args.results, 'r') as f:
        for line in f:
            r = json.loads(line)
            if r['speculative'] and '@2000' not in r['checkpoint']:
                continue
            entries.append(r)
    df = pd.DataFrame(entries)
    df = df[(df['model'] == 'mtp.models.stp.SingleTokenLM') | (df['ntoken'] == args.n_token)]
    df = df[df['ncomponent'].isin(args.n_components)]
    df = df[df['use_kv_cache'] == args.use_cache]

    def row_gen_setting(r: pd.Series) -> str:
        gen_setting = ''
        speculative = r['speculative']
        if speculative:
            gen_setting += 'Spec.'
            draft_top_p, target_top_p = r['draft_top_p'], r['target_top_p']
            argmax = draft_top_p == 0.0 and target_top_p == 0.0
            sampling = draft_top_p == 1.0 and target_top_p == 1.0
            if argmax:
                gen_setting += ' (argmax)'
            elif sampling:
                gen_setting += ' (sample)'
            else:
                gen_setting += f' (draft-p={draft_top_p}) (target-p={target_top_p})'
        else:
            gen_setting += 'Sampling'
        return gen_setting

    def row_model_id(r: pd.Series) -> str:
        model_id = ''
        model = r['model']
        if model == "mtp.models.stp.SingleTokenLM":
            model_id += 'STP'
        elif model == "mtp.models.mtp.MultiTokenLM":
            circuit = r['circuit']
            if circuit == 'fully_factorized':
                model_id += 'FF'
            else:
                n_component = r['ncomponent']
                if circuit == 'cp':
                    model_id += 'CP'
                elif circuit == 'hmm':
                    model_id += 'HMM'
                else:
                    assert False
                model_id += f' (r={n_component})'
        else:
            assert False
        return model_id

    df['gen_setting'] = df.apply(lambda r: row_gen_setting(r), axis=1)
    df['model_id'] = df.apply(lambda r: row_model_id(r), axis=1)

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.65)

    # Plot based on generation setting

    hue_order = ["STP", "FF", "CP (r=8)", "CP (r=32)"]
    ax = sb.barplot(
        df,
        x="gen_setting",
        y="tokens_per_second",
        hue="model_id",
        hue_order=hue_order
    )
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, fmt='{:.1f}')

    ax.set_xlabel("")
    ax.set_ylabel("Throughput (tok/s)")
    ax.grid(linestyle="--", which="major", alpha=0.4, linewidth=0.6)
    ax.grid(linestyle="--", which="minor", alpha=0.4, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), alignment="left")

    filename = f"throughput-{args.id}.pdf" if args.id else "throughput.pdf"
    plt.savefig(os.path.join("outputs", "plots", filename), bbox_inches='tight')
    plt.clf()
    plt.cla()

    # Plot number of accepted tokens per multi token model

    def row_accepted_tokens_xs(r: pd.Series) -> str:
        xs = r['hist_accepted_tokens'][0]
        return xs

    def row_accepted_tokens_probs(r: pd.Series) -> str:
        probs = r['hist_accepted_tokens'][1] / np.sum(r['hist_accepted_tokens'][1])
        return probs
    
    def row_model_id_acceptance_rate(r: pd.Series) -> str:
        model_id = r['model_id']
        rate = round(r['avg_accepted_tokens'], 1)
        #return model_id + " (\\mu_{\\text{accept}}=" + f"{rate})"
        return model_id + " (Avg.Rate=" + f"{rate})"

    df = df[df['model'] == 'mtp.models.mtp.MultiTokenLM']
    df = df[df['speculative'] == True]
    #df = df.sort_values(by=['ncomponent', 'circuit'], ascending=True)
    df['acceptance_xs'] = df.apply(lambda r: row_accepted_tokens_xs(r), axis=1)
    df['acceptance_probs'] = df.apply(lambda r: row_accepted_tokens_probs(r), axis=1)
    df['model_id_acceptance_rate'] = df.apply(lambda r: row_model_id_acceptance_rate(r), axis=1)
    df = df.drop('hist_accepted_tokens', axis=1)
    df = df.explode(['acceptance_xs', 'acceptance_probs'])

    setup_tueplots(1, 2, rel_width=1.5, hw_ratio=0.8, tight_layout=True)
    _, ax = plt.subplots(1, 2, sharey=True, squeeze=True)

    titles = ["Speculative (sample)", "Speculative (argmax)"]
    filters = [{'draft_top_p': 1.0, 'target_top_p': 1.0}, {'draft_top_p': 0.0, 'target_top_p': 0.0}]
    for i, title in zip(range(len(ax)), titles):
        df_ = df.copy()
        for k, v in filters[i].items():
            df_ = df_[df_[k] == v]
        hue_order_acceptance_rate = sorted(
            df_['model_id_acceptance_rate'].unique().tolist(),
            key=lambda miar: hue_order.index(
                ' '.join(miar.split(' ')[:2]) if 'CP' in miar else miar.split(' ')[0]
            )
        )
        sb.barplot(
            df_,
            width=0.8,
            x="acceptance_xs",
            y="acceptance_probs",
            hue="model_id_acceptance_rate",
            hue_order=hue_order_acceptance_rate,
            ax=ax[i]
        )
        ax[i].title.set_text(title)

        if i == 0:
            ax[i].set_ylabel("Probability of Acceptance")
        else:
            ax[i].set_ylabel("")
        ax[i].set_xlabel("Number of Draft Tokens")
        ax[i].grid(linestyle="--", which="major", alpha=0.4, linewidth=0.6)
        ax[i].grid(linestyle="--", which="minor", alpha=0.4, linewidth=0.6)
        ax[i].legend()

    filename = f"throughput-acceptance-{args.id}.pdf" if args.id else "throughput-acceptance.pdf"
    plt.savefig(os.path.join("outputs", "plots", filename), bbox_inches='tight')

    # unique_ntokens = np.aran
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
