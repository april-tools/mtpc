#!/usr/bin/env bash
# Run GPU memory profiling for all downloaded models using scripts/profile-v3-cli.py.

set -euo pipefail

source env.sh
. .venv/bin/activate

# No-LoRA models
sleep 10
python scripts/profile-v3-cli.py --timeout 0 --use-cache --disable-eos --num-tokens 1024   --adaptor none --circuits cp --n-tokens 8,16 --ranks 8,16,32,64,128   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-cp

sleep 10
python scripts/profile-v3-cli.py --timeout 0 --use-cache --disable-eos --num-tokens 1024   --adaptor none --circuits hmm --n-tokens 8,16 --ranks 32   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-hmm

sleep 10
python scripts/profile-v3-cli.py --timeout 0 --use-cache --disable-eos --num-tokens 1024   --adaptor none --circuits btree --n-tokens 8,16 --ranks 32   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-btree

sleep 10
python scripts/profile-v3-cli.py --timeout 0 --use-cache --disable-eos --num-tokens 1024   --adaptor none --circuits fully_factorized --n-tokens 8,16,32 --ranks 1 --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-ff

# LoRA-continued variants
for adaptor in lora-last-1 lora-last-2 lora-last-4; do
  sleep 10
  python scripts/profile-v3-cli.py --timeout 0 --use-cache --disable-eos --num-tokens 1024     --adaptor "$adaptor" --circuits btree --n-tokens 8,16 --ranks 32 --modes mtp,speculative --output-dir outputs/results/memory_usage/${adaptor}-btree

  sleep 10
  python scripts/profile-v3-cli.py --timeout 0 --use-cache --disable-eos --num-tokens 1024     --adaptor "$adaptor" --circuits fully_factorized --n-tokens 8,16 --ranks 1 --modes mtp,speculative --output-dir outputs/results/memory_usage/${adaptor}-ff
done
