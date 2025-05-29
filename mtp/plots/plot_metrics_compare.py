import re
import json
import argparse

import matplotlib.pyplot as plt

from mtp.plots.utils import setup_tueplots


def label_from_experiment(exp):
    pattern = r".*-(?P<circuit>\w+)-n-(?P<n>\d+)-r-(?P<r>\d+)"
    regex = re.compile(pattern)

    print(exp)
    m = regex.match(exp)
    if m:
        return f"{m.group('circuit')} n={m.group('n')} r={m.group('r')}"
    else:
        return "Unparseable-exp"


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

    n_cols = len(args.metrics)
    setup_tueplots(
        1,
        n_cols,
        rel_width=0.8 * n_cols,
        hw_ratio=0.8,
        tight_layout=True
    )
    fig, axes = plt.subplots(figsize=(12, 6), ncols=n_cols, nrows=1, sharey=True, sharex=True)


    for i, mname in enumerate(args.metrics):
        mpname = mname.replace('_', ' ')
    
        for experiment in args.experiments:
            # Metric Stats
            mc_stats = [row for row in mc_rows if row['expname'] == experiment]
            mc_stats = tuple(sorted(mc_stats, key=lambda x: x['step']))

            metric = tuple(row[mname] for row in mc_stats)
            steps = tuple(row['step'] for row in mc_stats)
            axes[i].plot(steps, metric, '-o', label=label_from_experiment(experiment))
            axes[i].set_title(mpname)
            axes[i].set_xlabel('# Train steps')
    axes[0].set_ylabel('Metric Value')
    axes[-1].legend()
    plt.suptitle('Validation Loss Over Training')
    plt.tight_layout()
    plt.show()
