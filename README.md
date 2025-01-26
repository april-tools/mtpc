# Overview:
This project contains our implementation of Multi-Token Prediction (MTP) with circuits.
The code is based on the [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).


# TODOS

* [ ] Check the circuit / sampling and parametrisation after latest changes.
* [ ] Check the architecture we use for MTP / consider alternatives.
* [ ] Train a default model and some MTP models.
* [ ] Implement approximate argmax prediction for circuit (to check if speculative decoding works, we need a way of decoding without randomness by using the circuit).
* [ ] Implement speculative decoding.
* [ ] Evaluate how well speculative decoding works - i.e. how many hits does the MTP model have when compared to the non-MTP model?
* [ ] Currently sum layers for all heads share the same params. Consider if we want to change this.

# Setup:

Download packages
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel psutil
pip install -r requirements.txt
```

Download data
```
python nanogpt/data/download.py 10
```

Change data paths in `nanogpt/configs/config.yaml` to your own paths.
Also specify :

1. The device IDs (comma separated) by setting `CUDA_VISIBLE_DEVICES`.
2. The MTP_ROOT environment variable; set it to the root directory of the project, see example in `env.sh`.


Run the training
```
source env.sh
torchrun --standalone --nproc_per_node=${GPUS} -m nanogpt.train
```

# Run Unit Tests

Running the tests can take up to one hour depending on the hardware.

```bash
export PYTHONPATH=.
pytest
```

# Experiments

## Throughput Evaluation

A first question is what generation throughput we can get with MTP - we measure this in tokens per sec (tps) using a batch size of one.

```bash
source env.sh
./bin/compute_throughput.sh
python -m plots.plot_throughput --device cuda --results results/throughput.jsonl
python -m plots.plot_throughput --device cpu --results results/throughput.jsonl
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

- Use Hydra everywhere (we can change the model via config using hydra.utils.instantiate).
- Added Script to compute and plot throughput for MTP vs Default model as we change ntokens.
