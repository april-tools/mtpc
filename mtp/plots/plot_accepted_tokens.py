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
    parser.add_argument('--filter-experiments', nargs='*', default=None,
                        help='Which experiments to keep')

    args = parser.parse_args()

    rows = []
    with open(args.results, 'r') as f:
        rows = []
        for line in f:
            row = json.loads(line)
            row['model'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            if args.filter_experiments is None or row['model'] in args.filter_experiments:
                rows.append(row)

    if args.type == 'accepted_tokens':
        fig, ax = plt.subplots(figsize=(10, 6), nrows=1, sharex=True)
        # fig, axes = plt.subplots(figsize=(10, 6), nrows=3)

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x['model'])):
            stats = tuple(sorted(stats, key=lambda x: x['step']))

            avg_accepted_tokens = tuple(row['avg_accepted_tokens'] for row in stats)
            steps = tuple(row['step'] for row in stats)
            ax.plot(steps, avg_accepted_tokens, '-o', label=model)

        ax.tick_params(axis='both')
        ax.set_ylabel('Average number of accepted tokens')
        ax.set_xlabel('# Training steps')
        ax.set_title('Token acceptance rate over training')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
    elif args.type == 'histogram':
        token_range = [0, 8]
        # token_range = tuple(range(ntoken + 1))
        ntoken = row['ntoken']
        fig, axes = plt.subplots(figsize=(6, 6), nrows=len(token_range))

        for i, (model, stats) in enumerate(groupby(rows, lambda x: x['model'])):
            stats = tuple(sorted(stats, key=lambda x: x['step']))
            for a, j in enumerate(token_range):
                counts = tuple(row['hist_accepted_tokens'][1][j] for row in stats)
                steps = tuple(row['step'] for row in stats)
                axes[a].plot(steps, counts, '-o', label='%s' % model)

        for a, j in enumerate(token_range):
            axes[a].set_title('# times %d token(s) generated' % (j + 1))

        axes[-1].set_xlabel('# Training steps')
        axes[-1].legend(loc='lower right')
        axes[1].set_ylabel('Number of generated tokens')
        plt.suptitle('Histogram of generated tokens over training')
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError('Unknown type option: %s' % args.type)
