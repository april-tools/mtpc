import json
import argparse

import matplotlib.pyplot as plt

from itertools import groupby


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Path to throughput.txt file (list of json).')
    parser.add_argument('--type', choices=['accepted_tokens', 'histogram'],
                        default='accepted_tokens', type=str,
                        help='Path to throughput.txt file (list of json).')
    parser.add_argument('--device', choices=('cuda', 'cpu'),
                        default='cuda', type=str,
                        help='Device to plot throughput for.')

    args = parser.parse_args()

    rows = []
    with open(args.results, 'r') as f:
        rows = []
        for line in f:
            row = json.loads(line)
            row['model'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            rows.append(row)

    if args.type == 'accepted_tokens':
        fig, ax = plt.subplots(figsize=(10, 6), nrows=1, sharex=True)
        # fig, axes = plt.subplots(figsize=(10, 6), nrows=3)

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x['model'])):
            stats = tuple(sorted(stats, key=lambda x: x['step']))

            avg_accepted_tokens = tuple(row['avg_accepted_tokens'] for row in stats)
            steps = tuple(row['step'] for row in stats)
            ax.plot(steps, avg_accepted_tokens, '-o', label=model)

        ax.set_ylabel('Average number of accepted tokens', fontsize=24)
        ax.set_xlabel('# Training steps', fontsize=24)
        ax.set_title('Token acceptance rate over training', fontsize=30)
        ax.legend(fontsize=20, loc='lower right')
        plt.tight_layout()
        plt.show()
    elif args.type == 'histogram':
        ntoken = row['ntoken']
        fig, axes = plt.subplots(figsize=(8, 10), nrows=ntoken + 1)

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x['model'])):
            stats = tuple(sorted(stats, key=lambda x: x['step']))
            for j in range(ntoken + 1):
                counts = tuple(row['hist_accepted_tokens'][1][j] for row in stats)
                steps = tuple(row['step'] for row in stats)
                axes[j].plot(steps, counts, '-o', label='%s' % model)

        for j in range(ntoken + 1):
            axes[j].set_title('# times %d token(s) generated' % (j + 1), fontsize=24)

        axes[-1].set_xlabel('# Training steps', fontsize=24)
        axes[-1].legend(fontsize=20, loc='lower right')
        axes[1].set_ylabel('Number of generated tokens (accepted tokens + 1)', fontsize=24)
        plt.suptitle('Histogram of generated tokens over training', fontsize=30)
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError('Unknown type option: %s' % args.type)
