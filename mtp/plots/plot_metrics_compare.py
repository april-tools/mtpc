import json
import argparse

import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-results', required=True,
                        help='Path to jsonl.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of experiment to plot results for.')
    parser.add_argument('--metrics', nargs='+', required=True,
                        help='List of metrics by name.')
    parser.add_argument('--device', choices=('cuda', 'cpu'),
                        default='cuda', type=str,
                        help='Device to plot throughput for.')

    args = parser.parse_args()

    with open(args.metric_results, 'r') as f:
        mc_rows = []
        for line in f:
            row = json.loads(line)
            row['expname'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            mc_rows.append(row)

    fig, axes = plt.subplots(figsize=(12, 9), nrows=len(args.metrics), sharex=True)

    # Metric Stats
    mc_stats = [row for row in mc_rows if row['expname'] == args.experiment]
    mc_stats = tuple(sorted(mc_stats, key=lambda x: x['step']))

    for i, mname in enumerate(args.metrics):
        mpname = mname.replace('_', ' ')

        metric = tuple(row[mname] for row in mc_stats)
        steps = tuple(row['step'] for row in mc_stats)
        axes[i].plot(steps, metric, '-o', label=mpname)
        axes[i].set_ylabel(mpname)
        axes[i].legend()
    axes[-1].set_xlabel('# Training steps', fontsize=24)

    plt.tight_layout()
    plt.show()
