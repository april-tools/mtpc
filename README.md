# Overview:
This is based on the [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

# Setup:
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel psutil
pip install -r requirements.txt
```

```
python data/cached_fineweb10B.py 10
```

Change paths in `nanogpt/configs/config.yaml` to your own paths.

# Code Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── attention.py      # Rotary, CausalSelfAttention
│   ├── mlp.py           # MLP, Block
│   └── gpt.py           # GPTConfig, GPT
├── data/
│   ├── __init__.py
│   └── dataloader.py    # DistributedDataLoader
├── config/
│   ├── __init__.py
│   └── hyperparameters.py  # Hyperparameters
├── utils/
│   ├── __init__.py
│   └── distributed.py    # DDP setup
└── train.py             # Main training loop
```
