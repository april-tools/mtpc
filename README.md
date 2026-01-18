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

## Download models
```
./bin/download_models
```

## Wandb

The training script is setup to use wandb to track metrics.
You will need to create an account and login to track metrics remotely.


# Running things


## Train Models:


### Smol (Start here)

#### Fit a NTP model
Fits a small transformer from scratch on Shakespeare char.
Useful for sanity checks as model trains in a few minutes.

```bash
# Train the default nanogpt model on shakespeare_char (see mtp/config/model/default.yaml)
torchrun --standalone \
	--nproc_per_node=1 \
	-m mtp.train \
	data=shakespeare_char \
	training=shakespeare_char  \
	model=stp  \
	lm.n_layer=4 \
	lm.n_head=4 \
	lm.n_embd=256 \
	lm.model.encoder_only=false \
	training.device_batch_size=128 \
	training.expname=my-smol-lm
```

Running the above will save the results (config and checkpoints) to a folder with the current date+time under `logs`.
Running the above should give:
```
[2025-05-22 16:23:29,775] - Setting up model... compile=True...
[2025-05-22 16:23:30,345] - Saving config and checkpoints to /disk/scratch/agrivas/nanoGPT/logs/2025-05-22/16-23-29...
[2025-05-22 16:23:30,346] - Save model: True...
[2025-05-22 16:23:30,346] - Save optimizer: True...
[2025-05-22 16:23:30,358] - Training on /disk/scratch/agrivas/nanoGPT/data/shakespeare_char/train.bin...
[2025-05-22 16:23:30,370] - Training DataLoader: total number of tokens: 1003854 across 1 files
[2025-05-22 16:23:30,370] - Validation DataLoader: total number of tokens: 111540 across 1 files
[2025-05-22 16:23:30,370] - During training we will see 524288000 tokens
[2025-05-22 16:23:30,370] - Each validation step will see 1048576 tokens
[2025-05-22 16:23:30,371] - step:0/2000 Saving model to /disk/scratch/agrivas/nanoGPT/logs/2025-05-22/16-23-29/model@0.pt...
[2025-05-22 16:23:42,167] - step:1/2000 val_loss:4.0616
[2025-05-22 16:23:42,168] - step:1/2000 train_loss:4.2427 lr:0.0004999997 time/step:9.80s
[2025-05-22 16:23:42,345] - step:2/2000 train_loss:4.0541 lr:0.0004999989 time/step:0.18s
...
[2025-05-22 16:25:12,954] - step:498/2000 train_loss:1.4269 lr:0.0004345981 time/step:0.18s
[2025-05-22 16:25:13,134] - step:499/2000 train_loss:1.4407 lr:0.0004343487 time/step:0.18s
[2025-05-22 16:25:13,540] - step:500/2000 val_loss:1.7431
[2025-05-22 16:25:13,605] - step:500/2000 Saved model to /disk/scratch/agrivas/nanoGPT/logs/2025-05-22/16-23-29/model@500.pt...
```

#### Distill NTP to MTP
Now, to distill the above NTP model into a MTP model, change `lm.model.from_checkpoint` below to point to your generated .pt checkpoint, and run:

```bash
# Train the mtp model on shakespeare_char (see mtp/config/model/mtp.yaml)
torchrun --standalone \
	--nproc_per_node=1 \
	-m mtp.train \
	data=shakespeare_char \
	training=shakespeare_char \
	model=mtp \
	model.beta=1 \ 
	model.gamma=.9 \
	model.kl_algorithm=full \
	circuit=cp \
	circuit.n_token=8 \
	circuit.n_component=8 \
	mt_head=transformer \
	lm.n_layer=4 \
	lm.n_head=4 \
	lm.n_embd=256 \
	lm.model.freeze=true \
	lm.model.lm=null \
	lm.model.encoder_only=false \
	training.save_model_every=100 \
	lm.model.from_checkpoint=logs/2025-05-22/16-23-29/model@500.pt \
	training.expname=my-smol-mtp-lm
```


```
[2025-05-22 16:27:23,446] - Setting up model... compile=True...
[2025-05-22 16:27:23,821] - Saving config and checkpoints to /disk/scratch/agrivas/nanoGPT/logs/2025-05-22/16-27-22...
[2025-05-22 16:27:23,821] - Save model: True...
[2025-05-22 16:27:23,821] - Save optimizer: True...
[2025-05-22 16:27:23,837] - Training on /disk/scratch/agrivas/nanoGPT/data/shakespeare_char/train.bin...
[2025-05-22 16:27:23,844] - Training DataLoader: total number of tokens: 1003854 across 1 files
[2025-05-22 16:27:23,844] - Validation DataLoader: total number of tokens: 111540 across 1 files
[2025-05-22 16:27:23,844] - During training we will see 524288000 tokens
[2025-05-22 16:27:23,844] - Each validation step will see 1048576 tokens
[2025-05-22 16:27:23,845] - step:0/2000 Saving model to /disk/scratch/agrivas/nanoGPT/logs/2025-05-22/16-27-22/model@0.pt...
/disk/scratch/agrivas/nanoGPT/.venv/lib/python3.10/site-packages/torch/autograd/graph.py:823: UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
grad.sizes() = [1, 256, 256], strides() = [256, 256, 1]
bucket_view.sizes() = [1, 256, 256], strides() = [65536, 256, 1] (Triggered internally at /pytorch/torch/csrc/distributed/c10d/reducer.cpp:327.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[2025-05-22 16:27:40,214] - step:1/2000 val_loss:1.9925
[2025-05-22 16:27:40,214] - step:1/2000 train_loss:2.2567 lr:0.0004999997 time/step:10.63s
[2025-05-22 16:27:43,508] - step:2/2000 train_loss:2.0322 lr:0.0004999989 time/step:3.29s
[2025-05-22 16:27:46,803] - step:3/2000 train_loss:1.8641 lr:0.0004999975 time/step:3.29s
...
[2025-05-22 16:44:21,832] - step:298/2000 train_loss:0.9015 lr:0.0004757964 time/step:3.31s
[2025-05-22 16:44:25,149] - step:299/2000 train_loss:0.9005 lr:0.0004756367 time/step:3.31s
[2025-05-22 16:44:31,798] - step:300/2000 val_loss:0.9477
[2025-05-22 16:44:31,878] - step:300/2000 Saved model to /disk/scratch/agrivas/nanoGPT/logs/2025-05-22/16-27-22/model@300.pt...
```

Logs acts like a draft folder, to save a model under a folder named `outputs/models/<dataset>/<expname>`), run:

```bash
./bin/save_experiment --experiments logs/*
```

where you can replace * by any path match.


#### Generate text from the Models

You can specify `--mode stp` to force single token prediction (even for mtp models).
For mtp models, use `--mode mtp` to generate `s` characters at a time and pass `--speculative` to enable speculative decoding.
You can also specify a prompt by using the `--prompt` parameter:

```bash
python -m mtp.generate --device cuda --checkpoint /path/to/stp/model@xxx.pt --mode stp --prompt ANTO --print
python -m mtp.generate --device cuda --checkpoint /path/to/mtp/model@xxx.pt --mode mtp --prompt ANTO --print
python -m mtp.generate --device cuda --checkpoint /path/to/mtp/model@xxx.pt --mode mtp --speculative --prompt ANTO --print
```


### Large

Here, instead of training our NTP LM from scratch, we take EvaByte-SFT which has been pretrained on a large corpus and fine-tuned on a data mix which includes Tulu 3.
#### Option 1: Distill EvaByte-SFT-NTP into MTP-CP by training on Tulu 3 using a cross-entropy loss.

```bash
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=tulu3-evabyte \
    training=tulu3-evabyte-long \
    lm=evabyte \
    model=mtp \
    circuit=cp \
    adaptor=lora-last-8 \
    mt_head=linear-evabyte \
    circuit.n_token=8 \
    circuit.n_component=8 \
    data.vocab_size=320 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
    training.device_batch_size=2 \
    training.expname=full-tulu-ce-evabyte-lora-last-8-cp-n-8-r-8
```

Our current acceptance rates and throughputs have been computed on models like the above trained for 2-3 days.
See [this script](scripts/evabyte-lora-tulu-2k/eval.sh) for details on the eval scripts.


#### Option 2: Distill EvaByte-SFT-NTP into MTP-CP by matching the NTP model via a KL loss.

```bash
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=tulu3-evabyte \
    training=tulu3-evabyte-long \
    lm=evabyte \
    model=mtp \
    circuit=cp \
    adaptor=lora-last-8 \
    mt_head=linear-evabyte \
    circuit.n_token=8 \
    circuit.n_component=2 \
    data.vocab_size=320 \
    model.model.beta=1 \
    model.model.gamma=0.9 \
	lm.model.encoder_only=false \
    training.device_batch_size=2 \
    training.expname=full-tulu-kl-evabyte-lora-last-8-cp-n-8-r-2
```


## Visualise Metrics

Assuming you have access to wandb, you can use the `plots.plot_wandb_metric` script to filter by wandb `--run-ids` and plot the metrics locally:

```bash
python mtp/plots/plot_wandb_metric.py --run-ids pd39py1e c8o44gf0 384rukjw --train-metrics --metrics ce_loss_at_2 ce_loss_at_4 ce_loss_at_6 ce_loss_at_8 --filepath outputs/plots/evabyte-tulu-2k/train.pdf --n-token 8 --smoothing .1 --share-y-all --n-rows 1
```


# Old Sections (below needs revision)


## Shakespeare Char-Level Model


### Throughput Evaluation (Untrained Models)

A first question is what generation throughput we can get with MTP.
We can get an upper bound on the throughput (without speculative decoding) even if we use an untrained model.
We measure throughput in tokens per sec (tps) using a batch size of one.

```bash
source env.sh
./bin/compute_throughput
python -m plots.plot_throughput --device cuda --results outputs/results/throughput.jsonl
```

The command above will append a json line of throughput stats for each model to `$MTP_ROOT/outputs/results/throughput_models.jsonl`

NOTE: tps will decrease as we increase the sequence length we are conditioning on, since the context increases.

### Train the models

To keep experiments in the example here fast, we train models on the `shakespeare_char` dataset.
```bash
./bin/train-shakespeare-char
```

### Plot the metrics

```bash
python -m plots.plot_wandb_metric --models autoregressive mtp-s=1-r=3 mtp-s=2-r=3 mtp-s=3-r=3 mtp-s=4-r=3 mtp-s=5-r=3 --metric valid/stp_loss --dataset shakespeare_char
```

### Throughput Evaluation (Trained Models)

While we keep track of validation metrics during training, we only compute those that we are tracking during training.
To get full validation results for checkpointed models, run:

#### Compute throughput for all models in a folder
```bash
./bin/validate_models /path/to/experiment/folder
```

The command above will append a json line of stats for each model to `$MTP_ROOT/outputs/results/throughput_models.jsonl`


### Metrics Evaluation (Trained Models)

While we keep track of validation metrics during training, we only compute those that we are tracking during training.
To get full validation results for checkpointed models, run:

#### Compute metrics for all models in a folder
```bash
./bin/validate_models /path/to/experiment/folder
```

The command above will append a json line of metrics for each model to `$MTP_ROOT/outputs/results/validate_models.jsonl`

#### Plot validation metrics
```bash
python mtp/plots/plot_metrics_compare.py --metric-results $MTP_ROOT/outputs/results/validate_models.jsonl --experiment your-expname --metrics kl_loss_at_1 kl_loss_ba_at_1
```
where:

* `--experiment`  is the experiment name you assigned via `training.expname` in the config or command line arguments.
* `--metrics` are metric we want to plot/compare, e.g. `--metrics kl_loss_at_1 ce_loss_at_1` to plot full kl vs cross-entropy for the next token

### Metric vs Throughput

Putting the above together, we have:

```bash
python mtp/plots/plot_throughput_vs_metric.py --metric-results $MTP_ROOT/outputs/results/validate_models.jsonl --throughput-results $MTP_ROOT/outputs/results/throughput_models.jsonl --experiment your-expname --metric kl_loss_at_1
```

where:

* `--experiment`  is the experiment name you assigned via `training.expname` in the config or command line arguments.
* `--metric` is the metric we want to plot, e.g. `--metric kl_loss_at_1` for full kl loss at 1.



# Notes

* While using `--mode stp --argmax` and `--mode mtp --speculative --argmax` with models of the same model family should generate the same output, quantised models may diverge between stp and mtp mode. One reason for this is that the transformer activations for the same input can be different if evaluated in a single forward pass, versus multiple forward passes one token at a time. This is especially true for quantised (bfloat16) models, see [this script for details](scripts/checks/test_multiple_vs_single.py).
