import argparse
import wandb

import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models by name.')
    parser.add_argument('--dataset', required=True,
                        help='Dataset the models are trained on.')
    parser.add_argument('--metric', required=True,
                        help='Which metric to plot.')

    args = parser.parse_args()

    wandb.login()
    api = wandb.Api()

    models = set(args.models)
    runs = [r for r in api.runs('mtp')
            if (r.config['expname'] in models)
            and (r.config['data']['name'] == args.dataset)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for r in runs:
        df = r.history(keys=['global_step', args.metric])

        step = df['global_step'].to_numpy()
        vals = df[args.metric].to_numpy()
        name = r.config['expname']
        ax.plot(step, vals, '-o', label=name)

    ax.set_ylim([0, None])
    ax.set_ylabel(args.metric, fontsize=24)
    ax.set_xlabel('# Training steps', fontsize=24)
    ax.set_title(f'{args.metric} for models on {args.dataset}', fontsize=30)
    ax.legend(fontsize=20, loc='lower right')
    plt.tight_layout()
    plt.show()
