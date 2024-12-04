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

# Notes

In traininglog.txt you can find my training logs.

Potential things to speed-up I haven't tried
- Tune LR, LR Schedule, BS
- Check betas = (0.8, 0.95) for Adam

Things to keep in mind:
- I lowered the number of eval tokens compared to original repo, therefore making it incomparable to the results from the original repo. One can increase it for the cost of longer execution.
