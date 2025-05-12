#!/bin/bash

python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-train-ns-250.jsonl --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-valid-ns-250.jsonl --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-train-ns-250.jsonl --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-valid-ns-250.jsonl --save

python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-train-ns-250.jsonl --type histogram --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/throughput-models-tulu-valid-ns-250.jsonl --type histogram --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-train-ns-250.jsonl --type histogram --save
python mtp/plots/plot_accepted_tokens.py --results scripts/evabyte-lora-tulu-2k/argmax_throughput_models-tulu-valid-ns-250.jsonl --type histogram --save
