import numpy as np
import argparse
import wandb

import matplotlib
import matplotlib.pyplot as plt

from mtp.plots.utils import setup_tueplots


def adapt_title(title):
    title = f'{metric}'.split('/')[-1]
    # title = title.replace('ce_loss_', '$\mathcal{L}$_')
    title = title.replace('ce_loss_', 'Cross-entropy_')
    title = title.replace('_at_', '@')
    return title


def exponential_moving_average(data, alpha=0.9):
    """Compute the exponential moving average."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Initialize with the first value

    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


"""
for n_token in 6 8 10 12;
do
  for n_component in 2 4;
  do
    python mtp/plots/plot_wandb_metric.py --dataset shakespeare_char --filepath cp-hmm-n-$n_token-r-$n_component-train.pdf \
      --models mtp-cp mtp-hmm --n-component $n_component --n-token $n_token \
      --train-metrics --metrics ce_loss_at --n-rows 3 --n-cols 4 --log-y
  done
done
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True, help='Dataset the models are trained on.')
    parser.add_argument('--models', nargs='+', default=None, help='List of models by name.')
    parser.add_argument('--run-ids', nargs='+', default=None, help='List of models by run id.')
    # parser.add_argument('--n-component', type=int, nargs='+', required=True, help='The number of components')
    parser.add_argument('--n-token', type=int, required=True, help='The number of tokens in MTP models')
    parser.add_argument('--smoothing', type=float, default=0., help='The smoothing value to apply')
    # parser.add_argument('--beta', type=float, default=0.0, help="The beta value weighting the KL term in the loss")
    # parser.add_argument('--gamma', type=float, default=0.9, help="The exponential discounting factor hyperparameter for each token loss")
    # parser.add_argument('--kl', type=str, choices=['full', 'binary_approx'], default='full', help='The kind of KL objective to use')
    parser.add_argument('--metrics', type=str, nargs='+', required=True, help='Which metrics to plot')
    parser.add_argument('--train-metrics', action='store_true', default=False, help='Whether to show training metrics instead of validation ones')
    # parser.add_argument('--freeze-lm', action='store_true', default=False, help="Whether to filter the results where the LLM is freezed")
    parser.add_argument('--n-rows', type=int, default=1, help='The number of plot rows')
    parser.add_argument('--n-cols', type=int, default=0, help='The number of plot cols. Defaults to the number of metrics provided.')
    parser.add_argument('--share-y-subplot', type=int, default=-1, help='The subplot column index on which to share the Y-axis of all subplots after it')
    parser.add_argument('--share-y-all', action='store_true', default=False, help='Whether to share the Y axis of all plots')
    parser.add_argument('--log-y', action='store_true', default=False, help='Whether to plot Ys in log space')
    parser.add_argument('--filepath', type=str, required=True, help='Where to store the figure')

    args = parser.parse_args()
    assert 0 <= args.smoothing <= 1
    assert (args.run_ids is None) or (args.models is None)

    wandb.login()
    api = wandb.Api()

    def run_filter_fn(r) -> bool:
        # if r.config['data']['name'] != args.dataset:
        #     return False
        expname = r.config['training'].get('expname', None)
        if expname not in args.models:
            return False
        # if r.config['model']['n_component'] not in args.n_component:
        #     return False
        # if r.config['model']['n_token'] != args.n_token:
        #     return False
        # if r.config['lm']['model']['freeze'] != args.freeze_lm:
        #     return False
        # if r.config['model']['beta'] != args.beta:
        #     return False
        # if r.config['model']['gamma'] != args.gamma:
        #     return False
        # if r.config['model']['beta'] != 0.0 and r.config['model']['model']['kl_algorithm'] != args.kl:
        #     return False
        return True

    if args.run_ids is not None:
        runs = api.runs(path='mtp', filters={"name": {"$in": args.run_ids}})
    else:
        runs = api.runs('mtp')
        # Filter experiment runs
        runs = list(filter(run_filter_fn, runs))
    print(f"Total number of runs: {len(runs)}")


    print(f"Number of filtered runs: {len(runs)}")
    if len(runs) == 0:
        print("No runs to plot, exiting ...")
        quit()

    # Set up plots
    n_rows = args.n_rows
    n_cols = len(args.metrics) if args.n_cols == 0 else args.n_cols
    setup_tueplots(
        n_rows,
        n_cols,
        rel_width=0.8 * n_cols,
        hw_ratio=0.8,
        tight_layout=True
    )
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=args.share_y_all, squeeze=False)

    metrics_grid = [[None] * n_cols for _ in range(n_rows)]
    # metrics = [args.metrics]
    metrics = args.metrics
    processed_metrics = []
    for metric in metrics:
        if len(metric) <= 3:
            processed_metrics.append(metric)
            continue
        if metric[-3:] != '_at':
            processed_metrics.append(metric)
            continue
        else:
            processed_metrics.append(metric)
        # for n in range(1, args.n_token + 1):
        #     processed_metrics.append(f'{metric}_{n}')

    assert len(processed_metrics) <= n_rows * n_cols
    for k, metric in enumerate(processed_metrics):
        i, j = k // n_cols, k % n_cols
        formatted_metric = f"{'train' if args.train_metrics else 'valid'}/{metric}"
        # formatted_metric = f"{metric}"
        metrics_grid[i][j] = formatted_metric

    def run_identifier(r) -> str:
        model = r.config['circuit']['name']
        entries = [model]
        entries.append(f"n={r.config['circuit']['n_token']}")
        entries.append(f"r={r.config['circuit']['n_component']}")
        beta = r.config['model']['beta']
        gamma = r.config['model']['gamma']
        # transf_tok_n_layer = r.config['model']['mt_head_hparams']['tok_transformer_n_layer']
        # transf_sum_n_layer = r.config['model']['mt_head_hparams']['sum_transformer_n_layer']
        # if transf_tok_n_layer != 0 or transf_sum_n_layer != 0:
        #     #transf_entries = []
        #     #if transf_tok_n_layer != 0:
        #     #    transf_entries.append(f't:{transf_tok_n_layer}')
        #     #if transf_sum_n_layer != 0:
        #     #    transf_entries.append(f's:{transf_sum_n_layer}')
        #     #entries.append(f"transf={'-'.join(transf_entries)}")
        #     entries.append(f"transf")
        # if beta != 0.0:
        #     entries.append(f"kl={r.config['model']['model']['kl_algorithm']}")
        #if gamma != 1.0:
        #    entries.append(f"dsc")
        return ' '.join(entries)

    for i in range(n_rows):
        for j in range(n_cols):
            metric = metrics_grid[i][j]
            if metric is None:
                ax[i][j].set_axis_off()
                continue
            ax[i][j].grid(linestyle="--", which="major", alpha=0.3, linewidth=0.6)
            ax[i][j].grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.4)
            ax[i][j].set_title(adapt_title(metric))
            for r in runs:
                df = r.history(keys=['global_step', metric])
                step = df['global_step'].to_numpy()
                vals = df[metric].to_numpy()
                label = run_identifier(r).replace('_', ' ')
                if args.smoothing == 0:
                    ax[i][j].plot(step, vals, '-', label=label, linewidth=1.8)
                else:
                    line = ax[i][j].plot(step, vals, '-', linewidth=1.8, alpha=0.1)
                    smooth_vals = exponential_moving_average(vals, alpha=args.smoothing)
                    ax[i][j].plot(step, smooth_vals, '-', label=label, linewidth=1.8, alpha=0.9, color=line[0].get_color())
            if 0 <= args.share_y_subplot < j:
                ax[i][j].sharey(ax[i][args.share_y_subplot])
                ax[i][j].tick_params(labelleft=False)
            if args.log_y:
                ax[i][j].set_yscale('log')
    ax[i][j].legend()

    # ax[0][-1].legend(loc='upper left', bbox_to_anchor=(1, 1), alignment='left')
    fig.savefig(args.filepath)
