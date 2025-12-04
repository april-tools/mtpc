# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Multi-Token Prediction (MTP) research project built on top of nanoGPT. The core idea is training models to predict multiple tokens simultaneously using probabilistic circuits (specifically CP decompositions) instead of traditional single-token autoregressive prediction.

**Key Concepts:**
- **NTP (Next Token Prediction)**: Standard autoregressive language models
- **MTP (Multi-Token Prediction)**: Models that predict multiple future tokens at once using structured probabilistic circuits
- **Circuits**: Probabilistic models (CP decomposition, HMM, binary trees) that capture dependencies between predicted tokens
- **Distillation**: Process of training an MTP model to match the predictions of a frozen NTP model

## Environment Setup

### Initial Installation

```bash
# Set CUDA_HOME first (required for flash-attn)
export CUDA_HOME=/opt/cuda-12.6.0  # Adjust to your CUDA path

# Using uv (recommended)
uv venv --python 3.10
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel psutil
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation

# Or using pip
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel psutil
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Environment Variables

**Always source env.sh before running any commands:**

```bash
source env.sh
```

This sets:
- `$MTP_ROOT`: Project root directory
- `$GPUS`: Number of GPUs (default: 1)
- `$CUDA_VISIBLE_DEVICES`: GPU device IDs
- `$WANDB_MODE`: Set to `disabled` or `online`
- `$CUBLAS_WORKSPACE_CONFIG`: For reproducibility
- `$HF_HOME`: HuggingFace cache location

## Common Development Commands

### Running Tests

```bash
# Set PYTHONPATH and run all tests (can take ~1 hour)
export PYTHONPATH=.
pytest

# Run specific test file
pytest tests/test_gpt.py

# Skip slow tests
pytest -m "not slow"

# Run single test
pytest tests/test_mtp.py::test_specific_function
```

### Training Models

**The training workflow is two-stage:**

1. **Stage 1: Train NTP baseline** (standard autoregressive model)
2. **Stage 2: Distill to MTP** (train MTP model from frozen NTP)

#### Quick Sanity Check (Shakespeare char-level)

Train a small NTP model (trains in minutes):

```bash
torchrun --standalone \
    --nproc_per_node=1 \
    -m mtp.train \
    data=shakespeare_char \
    training=shakespeare_char \
    model=stp \
    lm.n_layer=4 \
    lm.n_head=4 \
    lm.n_embd=256 \
    lm.model.encoder_only=false \
    training.device_batch_size=128 \
    training.expname=my-ntp-model
```

Distill to MTP (update checkpoint path to your trained model):

```bash
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
    lm.model.from_checkpoint=logs/YYYY-MM-DD/HH-MM-SS/model@500.pt \
    training.expname=my-mtp-model
```

#### Large-Scale Training (EvaByte on Tulu3)

Using cross-entropy loss:

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
    training.expname=my-evabyte-mtp
```

### Text Generation

```bash
# Single token prediction (NTP)
python -m mtp.generate \
    --device cuda \
    --checkpoint /path/to/model@xxx.pt \
    --mode stp \
    --prompt "HELLO" \
    --print

# Multi-token prediction (MTP)
python -m mtp.generate \
    --device cuda \
    --checkpoint /path/to/mtp/model@xxx.pt \
    --mode mtp \
    --prompt "HELLO" \
    --print

# MTP with speculative decoding
python -m mtp.generate \
    --device cuda \
    --checkpoint /path/to/mtp/model@xxx.pt \
    --mode mtp \
    --speculative \
    --prompt "HELLO" \
    --print
```

### Data Management

```bash
# Download FineWeb data (10 chunks)
./bin/download_data --numchunks 10 --dataset fineweb

# Download all FineWeb-edu chunks
./bin/download_data --dataset fineweb-edu

# Download pretrained models
./bin/download_models
```

### Model Validation & Evaluation

```bash
# Validate models in a folder
./bin/validate_models /path/to/experiment/folder

# This appends results to:
# - $MTP_ROOT/outputs/results/validate_models.jsonl (metrics)
# - $MTP_ROOT/outputs/results/throughput_models.jsonl (throughput)
```

### Saving Experiments

Training outputs go to `logs/` (temporary). To save permanently:

```bash
# Save experiment to outputs/models/<dataset>/<expname>
./bin/save_experiment --experiments logs/YYYY-MM-DD/HH-MM-SS
```

### Plotting and Analysis

```bash
# Plot training metrics from wandb
python mtp/plots/plot_wandb_metric.py \
    --run-ids <wandb-run-id1> <wandb-run-id2> \
    --train-metrics \
    --metrics ce_loss_at_2 ce_loss_at_4 \
    --filepath outputs/plots/my-plot.pdf \
    --n-token 8 \
    --smoothing .1

# Plot validation metrics comparison
python mtp/plots/plot_metrics_compare.py \
    --metric-results outputs/results/validate_models.jsonl \
    --experiment my-expname \
    --metrics kl_loss_at_1 ce_loss_at_1

# Plot throughput vs metric tradeoff
python mtp/plots/plot_throughput_vs_metric.py \
    --metric-results outputs/results/validate_models.jsonl \
    --throughput-results outputs/results/throughput_models.jsonl \
    --experiment my-expname \
    --metric kl_loss_at_1
```

## Architecture Overview

### Core Components

The codebase is organized into three main model types:

1. **STP (Single Token Prediction)** - `mtp/models/stp.py`
   - Standard autoregressive language model wrapper
   - Used for baseline training

2. **MTP (Multi-Token Prediction)** - `mtp/models/mtp.py`
   - Composed of three parts:
     - **LM encoder**: Provides contextual embeddings (can be frozen)
     - **mt_head**: Expands embeddings into circuit parameters
     - **Circuit**: Probabilistic model over multiple output tokens

3. **Circuits** - `mtp/models/circuits.py`
   - Implements structured probabilistic models:
     - `cp`: CP (CANDECOMP/PARAFAC) decomposition - tensor factorization
     - `hmm`: Hidden Markov Model structure
     - `btree`: Binary tree factorization
     - `random-btree`: Randomized binary tree
   - Uses the `cirkit` library for probabilistic circuit operations

### Configuration System (Hydra)

All configuration is managed via Hydra YAML configs in `configs/`:

- `configs/config.yaml`: Main config with defaults
- `configs/model/`: Model architectures (stp, mtp, mtp-cp)
- `configs/lm/`: Language model configs (nanogpt, evabyte)
- `configs/circuit/`: Circuit structures (cp, hmm, btree)
- `configs/mt_head/`: Multi-token head architectures
- `configs/data/`: Dataset configurations
- `configs/training/`: Training hyperparameters
- `configs/adaptor/`: LoRA and adaptation configs

**Override configs on command line:**

```bash
# Override nested config values
torchrun -m mtp.train \
    lm=evabyte \
    lm.n_layer=8 \
    training.learning_rate=0.001 \
    model.beta=0.5
```

### Key Model Parameters

When working with MTP models, understand these critical parameters:

- **`n_token`**: Number of tokens to predict simultaneously (e.g., 8)
- **`n_component`**: Rank of the CP decomposition (controls circuit complexity)
- **`beta`**: KL vs CE loss weight (0=CE only, 1=KL only, 0.5=balanced)
- **`gamma`**: Discount factor for future tokens (1=equal weight, <1=prefer near tokens)
- **`kl_algorithm`**: Type of KL computation (`full` or `binary_approx`)

### Data Loading

Two main data loader types in `mtp/data/`:

1. **LocalDataLoader** - For `.bin` files (e.g., Shakespeare)
2. **HFDataLoader** - For HuggingFace datasets

Data is loaded via `DistributedDataLoader.resolve()` which selects the appropriate loader based on file type.

### Loss Computation

Loss functions in `mtp/models/loss.py`:

- **Cross-Entropy (CE)**: Standard token-level loss
- **Full KL**: Complete KL divergence between NTP and MTP distributions
- **Binary Approx KL**: Efficient approximation of KL using binary factorization

The MTP model computes weighted combinations of these based on `beta` parameter.

### LoRA Integration

For large models, LoRA adaptation is used via `mtp/models/lora_split_lm.py`:

- Adds low-rank adapters to specific layers
- Config: `configs/adaptor/lora-last-8.yaml` (adapts last 8 layers)
- Allows efficient fine-tuning of pretrained models

## Important Development Notes

### Checkpoint Management

- Training creates checkpoints in `logs/YYYY-MM-DD/HH-MM-SS/`
- Checkpoints are named `model@<step>.pt`
- Config is saved alongside as `config.yaml`
- Use `./bin/save_experiment` to move from logs to permanent storage

### Distributed Training

All training uses `torchrun` even for single GPU:

```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 -m mtp.train ...

# Multi-GPU (set $GPUS in env.sh first)
torchrun --standalone --nproc_per_node=$GPUS -m mtp.train ...
```

### Model Compilation

By default, models are compiled with `torch.compile`:
- Set `compile=true` in config (default)
- Adds compilation overhead but speeds up training
- Can disable with `compile=false` for debugging

### Reproducibility

For deterministic results:
1. Set random seed via `training.random_seed` in config
2. Source `env.sh` (sets `CUBLAS_WORKSPACE_CONFIG`)
3. Training code uses `set_deterministic()` function

### Wandb Tracking

Training metrics are logged to wandb:
- Control with `$WANDB_MODE` environment variable
- Set to `disabled` for no logging (default in env.sh)
- Set to `online` to enable tracking
- Run IDs are used for plotting: `python mtp/plots/plot_wandb_metric.py --run-ids <id>`

## Testing Notes

Tests in `tests/` directory cover:
- `test_gpt.py`: GPT model components
- `test_mtp.py`: Multi-token prediction logic
- `test_circuit_model.py`: Circuit construction and queries
- `test_data.py`: Data loading
- `test_loss.py`: Loss computation
- `test_packing.py`: Sequence packing utilities

Mark slow tests with `@pytest.mark.slow` decorator.

## Special Files

- `env.sh`: Environment setup (always source before running commands)
- `memory.sh`: Memory profiling script
- `pytest.ini`: Test configuration
- `bin/`: Executable scripts for common operations
- `scripts/`: Analysis and evaluation scripts
- `tools/`: Utility scripts
