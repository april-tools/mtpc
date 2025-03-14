# Overview:
This project contains our implementation of Multi-Token Prediction (MTP) with circuits.
The code is based on the [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).


# Setup:

## Download code

```bash
git clone git@github.com:PiotrNawrot/nanoGPT.git && cd nanoGPT
```

### Prepare package installation
For flash-attn build to work, set the `CUDA_HOME` env variable to point to your CUDA path, e.g.:

```
export CUDA_HOME=/opt/cuda-12.6.0
```

### Environment installation using uv
```
uv venv --python 3.10
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel psutil
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
```

### Environment installation using pip
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel psutil
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Environment Variables

All paths are configured w.r.t. the project root folder, `$MTP_ROOT`.
To set it, run the following command from the root folder:

```bash
# From root directory of the project run the following, it sets $MTP_ROOT
source env.sh
```

You may want to adapt/change:

1. The number of GPUs and device IDs (comma separated) by setting `CUDA_VISIBLE_DEVICES`.
2. Whether to use wandb or not (currently `disabled`, change to `online` for logging)


# Run Unit Tests

Running the tests can take up to one hour depending on the hardware.

```bash
export PYTHONPATH=.
pytest
```

## Download data
```
# Download first 10 chunks of fineweb train dataset
./bin/download_data --numchunks 10 --dataset fineweb
# Download all chunks of fineweb-edu train dataset
./bin/download_data --dataset fineweb-edu
```

## Wandb

The training script is setup to use wandb to track metrics.
You will need to create an account to track metrics remotely.


# Running things


## Train Models:

```bash
# Train the default nanogpt model on shakespeare_char (see mtp/config/model/default.yaml)
torchrun --standalone --nproc_per_node=${GPUS} -m mtp.train data=shakespeare_char model=default model.n_embd=384
# Train the mtp model on shakespeare_char (see mtp/config/model/mtp.yaml)
torchrun --standalone --nproc_per_node=${GPUS} -m mtp.train data=shakespeare_char model=mtp model.n_embd=384 model.n_token=3 model.n_component=5
```

The outputs of each experiment (config and checkpoints) are written to a folder with the current date+time under `logs`.
If you want to keep the model (i.e. move it to `outputs/models/<dataset>/expname`), run:

```bash
./bin/save_experiment --experiments logs/date/time/*
```

## Visualise Metrics

Assuming you have access to wandb, you can use the `plots.plot_metric` script to filter by dataset and `expname` (models) and plot the metrics locally:

```bash
python -m plots.plot_metric --models autoregressive mtp-s=3-r=5 --metric valid/stp_loss --dataset shakespeare_char
```

## Generate from Models:

You can specify `--mode stp` to force single token prediction (even for mtp models).
For mtp models, use `--mode mtp` to generate `s` characters at a time.
You can also specify a prompt by using the `--prompt` parameter:


```bash
python -m mtp.generate --device cuda --checkpoint /path/to/mtp/model@xxx.pth --mode mtp --prompt ANTO
python -m mtp.generate --device cuda --checkpoint /path/to/stp/model@xxx.pth --mode stp --prompt ANTO
```


# Experiments


## Shakespeare Char-Level Model


### Train the models

As a sanity check, we train models on the `shakespeare_char` dataset.
```bash
./bin/train-shakespeare-char
```

### Plot the metrics

```bash
python -m plots.plot_metric --models autoregressive mtp-s=1-r=3 mtp-s=2-r=3 mtp-s=3-r=3 mtp-s=4-r=3 mtp-s=5-r=3 --metric valid/stp_loss --dataset shakespeare_char
```

### Download the models

The trained models can be downloaded via:

```bash
./bin/download_models
```

### Throughput Evaluation

A first question is what generation throughput we can get with MTP - we measure this in tokens per sec (tps) using a batch size of one.

```bash
source env.sh
./bin/compute_throughput
python -m plots.plot_throughput --device cuda --results outputs/results/throughput.jsonl
python -m plots.plot_throughput --device cpu --results outputs/results/throughput.jsonl
```

NOTE: tps will decrease as we increase the sequence length we are conditioning on: since the context increases.



# Notes

In traininglog.txt you can find my training logs.

Potential things to speed-up I haven't tried
- Tune LR, LR Schedule, BS
- Check betas = (0.8, 0.95) for Adam

Things to keep in mind:
- I lowered the number of eval tokens compared to original repo, therefore making it incomparable to the results from the original repo. One can increase it for the cost of longer execution.


# Changes

- Added scripts to train and check throughput of shakespeare_char models.
- Monitor experiments using wandb.
- Use Hydra everywhere (we can change the model via config using hydra.utils.instantiate).
- Added Script to compute and plot throughput for MTP vs Default model as we change ntokens.
- Adapted Scripts to train a character level model for sanity check
- Serialised config to logs output dir and the model every eval iterations
