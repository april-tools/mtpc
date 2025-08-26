import os
import json
import argparse
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from itertools import groupby
from mtp.plots.utils import setup_tueplots


def get_label(stats):
    n = stats[-1]['ntoken']
    r = stats[-1]['ncomponent']
    circuit = stats[-1]['circuit'].replace('_', '-').upper()
    return f"{circuit} r={r:<2} n={n:<2}"


def get_model_type(model):
    if '-cp-' in model:
        return 'cp'
    if '-hmm-' in model:
        return 'hmm'
    else:
        return 'unknown'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='Path to throughput.txt file (list of json).')
    parser.add_argument('--type', choices=['accepted_tokens', 'histogram'],
                        default='accepted_tokens', type=str,
                        help='Path to throughput.txt file (list of json).')
    parser.add_argument('--device', choices=('cuda', 'cpu'),
                        default='cuda', type=str,
                        help='Device to plot throughput for.')
    parser.add_argument('--ntokens', type=int, nargs='+', help="The number of tokens for the circuit model")
    parser.add_argument('--ncomponents', type=int, nargs='+', help="The number of components for the circuit model")
    parser.add_argument('--decoding', type=str, required=True, choices=("argmax", "sampling"), help="What decoding to use")
    parser.add_argument('--steps', type=int, nargs='+', default=None, help="What step to filter for")
    parser.add_argument('--save', action='store_true',
                        help='If specified saves the plot instead of interactive plot.')
    parser.add_argument('--id', type=str, default="", help="The id of the experiment that will be appended to the filename")
    parser.add_argument('--filter-experiments', nargs='+', default=None,
                        help='Which experiments to keep')

    args = parser.parse_args()

    rows = []
    with open(args.results, 'r') as f:
        rows = []
        for line in f:
            row = json.loads(line)
            row['model'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            row['model_type'] = get_model_type(row['model'])
            if args.filter_experiments is not None and row['model'] not in args.filter_experiments:
                continue
            if row['ntoken'] not in args.ntokens:
                continue
            if row['ncomponent'] not in args.ncomponents:
                continue
            if args.steps is not None:
                if row['step'] not in args.steps:
                    continue
            if row['argmax'] != (args.decoding == 'argmax'):
                continue
            rows.append(row)

    rows = tuple(sorted(rows, key=lambda x: (x['model_type'], x['ncomponent'])))

    setup_tueplots(1, 1, rel_width=1.0, hw_ratio=0.8)


    if args.type == 'accepted_tokens':
        fig, ax = plt.subplots(figsize=(10, 6), nrows=1, sharex=True)
        # fig, axes = plt.subplots(figsize=(10, 6), nrows=3)

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x['model'])):
            stats = tuple(sorted(stats, key=lambda x: x['step']))

            means, stds, unique_steps = [], [], []
            for step, iter_stats in groupby(stats, lambda x: x['step']):
                runs = []
                for run in iter_stats:
                    runs.append(run['avg_accepted_tokens'])
                runs = np.array(runs)
                means.append(np.mean(runs))
                stds.append(np.std(runs))
                unique_steps.append(step)

            means, stds = np.array(means), np.array(stds)

            avg_accepted_tokens = tuple(row['avg_accepted_tokens'] for row in stats)
            steps = tuple(row['step'] for row in stats)

            label = get_label(stats)

            mean_scatter = ax.scatter(unique_steps, means, label=label, s=40, alpha=.9)
            color = mean_scatter.get_facecolor()

            mean_plot = ax.plot(unique_steps, means, '-', alpha=.9)
            ax.fill_between(unique_steps, means-stds, means+stds, color=color, alpha=.1)

            scatter = ax.scatter(steps, avg_accepted_tokens, color=color, s=10, alpha=.3)

        ax.tick_params(axis='both')
        ax.set_ylabel('Mean Accepted Tokens')
        ax.set_xlabel('# Training steps')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), alignment='left')
    elif args.type == 'histogram':
        token_range = [0, 8]
        # token_range = tuple(range(ntoken + 1))
        ntoken = row['ntoken']
        fig, axes = plt.subplots(figsize=(9, 11), nrows=len(token_range))

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x['model'])):
            stats = tuple(sorted(stats, key=lambda x: x['step']))
            for a, j in enumerate(token_range):
                counts = tuple(row['hist_accepted_tokens'][1][j] for row in stats)
                steps = tuple(row['step'] for row in stats)
                axes[a].plot(steps, counts, '-o', label=get_label(stats))

        for a, j in enumerate(token_range):
            axes[a].set_title('# times %d token(s) generated' % (j + 1))

        axes[-1].set_xlabel('# Training steps')
        axes[-1].legend(loc='best')
        fig.supylabel('Number of generated tokens')
        plt.suptitle('Histogram of generated tokens over training')
    else:
        raise ValueError('Unknown type option: %s' % args.type)
    if args.save:
        filename = os.path.basename(args.results).replace('.jsonl', '')
        save_path = os.path.join("outputs", "plots", f"{args.type.replace('_', '-')}-{args.id}.pdf")
        # Save a pdf
        print(f"Saving plot to {save_path} ...")
        plt.savefig(save_path, bbox_inches='tight')
        # Also save a png
        save_path = os.path.join("outputs", "plots", f"{args.type.replace('_', '-')}-{args.id}.png")
        print(f"Also, saving plot to {save_path} ...")
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
