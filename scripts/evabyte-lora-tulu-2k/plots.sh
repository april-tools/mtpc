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


## Validation plots
python mtp/plots/plot_metrics_compare.py  --metric-results scripts/evabyte-lora-tulu-2k/validate_models.jsonl       --experiments  full-tulu-evabyte-lora-last-8-cp-n-8-r-8 full-tulu-evabyte-lora-last-8-cp-n-8-r-32      --metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_7 kl_loss_at_8
python mtp/plots/plot_metrics_compare.py  --metric-results scripts/evabyte-lora-tulu-2k/validate_models.jsonl       --experiments  full-tulu-evabyte-lora-last-8-cp-n-8-r-8 full-tulu-evabyte-lora-last-8-cp-n-8-r-32      --metrics ce_loss_at_1 ce_loss_at_2 ce_loss_at_3 ce_loss_at_7 ce_loss_at_8
python mtp/plots/plot_metrics_compare.py  --metric-results scripts/evabyte-lora-tulu-2k/validate_models.jsonl       --experiments  full-tulu-evabyte-lora-last-8-cp-n-8-r-8 full-tulu-evabyte-lora-last-8-cp-n-8-r-32      --metrics kl_loss_at_1 ce_loss_at_1
