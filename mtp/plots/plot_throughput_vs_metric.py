import json
import argparse

import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-results', required=True,
                        help='Path to jsonl.')
    parser.add_argument('--throughput-results', required=True,
                        help='Path to jsonl.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Name of experiment to plot results for.')
    parser.add_argument('--metric', type=str, required=True,
                        help='Which metric to plot.')
    parser.add_argument('--device', choices=('cuda', 'cpu'),
                        default='cuda', type=str,
                        help='Device to plot throughput for.')

    args = parser.parse_args()

    with open(args.throughput_results, 'r') as f:
        tp_rows = []
        for line in f:
            row = json.loads(line)
            row['expname'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            tp_rows.append(row)

    with open(args.metric_results, 'r') as f:
        mc_rows = []
        for line in f:
            row = json.loads(line)
            row['expname'], step = row['checkpoint'].split('@')
            row['step'] = int(step)
            mc_rows.append(row)

    fig, (ax, ax2) = plt.subplots(figsize=(12, 9), nrows=2, sharex=True)


    # Throughput Stats
    tp_stats = [row for row in tp_rows if row['expname'] == args.experiment]
    tp_stats = tuple(sorted(tp_stats, key=lambda x: x['step']))

    avg_accepted_tokens = tuple(row['avg_accepted_tokens'] for row in tp_stats)
    steps = tuple(row['step'] for row in tp_stats)
    ax.plot(steps, avg_accepted_tokens, '-o', label=args.experiment)

    ax.set_ylabel('Avg accepted tokens', fontsize=24)
    ax.set_title('Throughput vs Metric over training', fontsize=30)
    # ax.legend(fontsize=20, loc='lower right')

    # Metric Stats
    mc_stats = [row for row in mc_rows if row['expname'] == args.experiment]
    mc_stats = tuple(sorted(mc_stats, key=lambda x: x['step']))

    metric = tuple(row[args.metric] for row in mc_stats)
    steps = tuple(row['step'] for row in mc_stats)
    ax2.plot(steps, metric, '-o', label=args.experiment)
    ax2.set_ylabel(args.metric.replace('_', ' '))
    ax2.set_xlabel('# Training steps', fontsize=24)

    plt.tight_layout()
    plt.show()
