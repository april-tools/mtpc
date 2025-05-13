#!/bin/bash

python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-train-ns-250.jsonl --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-valid-ns-250.jsonl --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-train-ns-250.jsonl --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-valid-ns-250.jsonl --save

python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-train-ns-250.jsonl --type histogram --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-valid-ns-250.jsonl --type histogram --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-train-ns-250.jsonl --type histogram --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-valid-ns-250.jsonl --type histogram --save


## Training plots
python mtp/plots/plot_wandb_metric.py --run-ids pd39py1e c8o44gf0 384rukjw --train-metrics --metrics ce_loss_at_2 ce_loss_at_4 ce_loss_at_6 ce_loss_at_8 --filepath outputs/plots/evabyte-tulu-2k/train.pdf --n-token 8 --smoothing .1 --share-y-all --n-rows 1
