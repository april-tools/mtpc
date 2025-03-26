import argparse
import wandb

import matplotlib.pyplot as plt

from mtp.plots.utils import setup_tueplots

"""
python mtp/plots/plot_wandb_metric.py --dataset shakespeare_char --filepath cp-hmm-n-4-r-4-valid.pdf --models mtp-cp mtp-hmm --n-component 4 --n-token 4 --metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_4 ce_loss_at_1 ce_loss_at_2 ce_loss_at_3 ce_loss_at_4 loss --n-rows 3 --n-cols 4 --share-y-subplot 1 --log-y
python mtp/plots/plot_wandb_metric.py --dataset shakespeare_char --filepath cp-hmm-n-4-r-4-train.pdf --train-metrics --models mtp-cp mtp-hmm --n-component 4 --n-token 4 --metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_4 ce_loss_at_1 ce_loss_at_2 ce_loss_at_3 ce_loss_at_4 loss --n-rows 3 --n-cols 4 --share-y-subplot 1 --log-y
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset the models are trained on.')
    parser.add_argument('--models', nargs='+', required=True, help='List of models by name.')
    parser.add_argument('--n-component', type=int, required=True, help='The number of components')
    parser.add_argument('--n-token', type=int, required=True, help='The number of tokens in MTP models')
    parser.add_argument('--kl', type=str, choices=['full', 'binary_approx'], default='full', help='The kind of KL objective to use')
    parser.add_argument('--metrics', nargs='+', required=True, help='Which metrics to plot')
    parser.add_argument('--train-metrics', action='store_true', default=False, help='Whether to show training metrics instead of validation ones')
    parser.add_argument('--n-rows', type=int, default=1, help='The number of plot rows')
    parser.add_argument('--n-cols', type=int, default=0, help='The number of plot cols. Defaults to the number of metrics provided.')
    parser.add_argument('--share-y-subplot', type=int, default=-1, help='The subplot column index on which to share the Y-axis of all subplots after it')
    parser.add_argument('--log-y', action='store_true', default=False, help='Whether to plot Ys in log space')
    parser.add_argument('--filepath', type=str, required=True, help='Where to store the figure')

    args = parser.parse_args()

    wandb.login()
    api = wandb.Api()
    runs = api.runs('mtp')
    print(f"Total number of runs: {len(runs)}")

    def run_filter_fn(r) -> bool:
        if r.config['data']['name'] != args.dataset:
            return False
        if r.config['model']['name'] not in args.models:
            return False
        if r.config['model']['n_component'] != args.n_component:
            return False
        if r.config['model']['n_token'] != args.n_token:
            return False
        if r.config['model']['model']['kl_algorithm'] != args.kl:
            return False
        return True

    # Filter experiment runs
    runs = list(filter(run_filter_fn, runs))
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
        hw_ratio=0.75,
        tight_layout=True
    )
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False)

    metrics_grid = [[None] * n_cols for _ in range(n_rows)]
    assert len(args.metrics) <= n_rows * n_cols
    for k, metric in enumerate(args.metrics):
        i, j = k // n_cols, k % n_cols
        metrics_grid[i][j] = f"{'train' if args.train_metrics else 'valid'}/{metric}"

    def run_identifier(r) -> str:
        model = r.config['model']['name'].split('-')[1]
        n_component = r.config['model']['n_component']
        n_token = r.config['model']['n_token']
        kl_algorithm = r.config['model']['model']['kl_algorithm']
        transf_tok_n_layer = r.config['model']['mt_head_hparams']['tok_transformer_n_layer']
        transf_sum_n_layer = r.config['model']['mt_head_hparams']['sum_transformer_n_layer']
        if transf_tok_n_layer == 0 and transf_sum_n_layer == 0:
            return f'{model} n={n_token} r={n_component} kl={kl_algorithm}'
        return f'{model} n={n_token} r={n_component} kl={kl_algorithm} tf=t{transf_tok_n_layer}-s{transf_sum_n_layer}'

    for i in range(n_rows):
        for j in range(n_cols):
            metric = metrics_grid[i][j]
            if metric is None:
                ax[i][j].set_axis_off()
                continue
            ax[i][j].grid(linestyle="--", which="major", alpha=0.3, linewidth=0.6)
            ax[i][j].grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.4)
            ax[i][j].set_title(f'{metric}')
            for r in runs:
                df = r.history(keys=['global_step', metric])
                step = df['global_step'].to_numpy()
                vals = df[metric].to_numpy()
                label = run_identifier(r)
                ax[i][j].plot(step, vals, '-', label=label, linewidth=1.8)
            if 0 <= args.share_y_subplot < j:
                ax[i][j].sharey(ax[i][args.share_y_subplot])
                ax[i][j].tick_params(labelleft=False)
            if args.log_y:
                ax[i][j].set_yscale('log')
            ax[i][j].legend()

    # ax[0][-1].legend(loc='upper left', bbox_to_anchor=(1, 1), alignment='left')
    fig.savefig(args.filepath)
