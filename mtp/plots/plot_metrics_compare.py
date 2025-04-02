import re
import json
import argparse

import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-results', required=True,
                        help='Path to jsonl.')
    parser.add_argument('--experiments', nargs='+', required=True,
                        help='List of metrics by name.')
    parser.add_argument('--metrics', nargs='+', required=True,
                        help='List of metrics by name.')

    args = parser.parse_args()

    with open(args.metric_results, 'r') as f:
        mc_rows = []
        for line in f:
            row = json.loads(line)
            row['expname'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            mc_rows.append(row)

    fig, axes = plt.subplots(figsize=(16, 8), ncols=len(args.metrics), nrows=1)


    for i, mname in enumerate(args.metrics):
        mpname = mname.replace('_', ' ')
    
        for experiment in args.experiments:
            # Metric Stats
            mc_stats = [row for row in mc_rows if row['expname'] == experiment]
            mc_stats = tuple(sorted(mc_stats, key=lambda x: x['step']))

            metric = tuple(row[mname] for row in mc_stats)
            steps = tuple(row['step'] for row in mc_stats)
            axes[i].plot(steps, metric, '-o', label=experiment)
            axes[i].set_ylabel(mpname, fontsize=20)
            axes[i].set_xlabel('# Training steps', fontsize=24)
    axes[-1].legend(fontsize=12)
    plt.suptitle('Validation Loss Over Training', fontsize=30)
    plt.tight_layout()
    plt.show()
