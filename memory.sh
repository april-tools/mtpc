#!/usr/bin/env bash
# Run GPU memory profiling for all downloaded models using scripts/profile-cli.py.

set -euo pipefail

source env.sh
. .venv/bin/activate

# No-LoRA models
python scripts/profile-cli.py --use-cache --disable-eos --num-tokens 128   --adaptor none --circuits cp --n-tokens 8,16 --ranks 8,16,32,64,128   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-cp

python scripts/profile-cli.py --use-cache --disable-eos --num-tokens 128   --adaptor none --circuits hmm --n-tokens 8,16 --ranks 32   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-hmm

python scripts/profile-cli.py --use-cache --disable-eos --num-tokens 128   --adaptor none --circuits btree --n-tokens 8,16 --ranks 32   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-btree

python scripts/profile-cli.py --use-cache --disable-eos --num-tokens 128   --adaptor none --circuits fully_factorized --n-tokens 8,16,32 --ranks 1   --modes mtp,speculative --output-dir outputs/results/memory_usage/no-lora-ff

# LoRA-continued variants
for adaptor in lora-last-1 lora-last-2 lora-last-4; do
  python scripts/profile-cli.py --use-cache --disable-eos --num-tokens 128     --adaptor "$adaptor" --circuits btree --n-tokens 8,16 --ranks 32     --modes mtp,speculative --output-dir outputs/results/memory_usage/${adaptor}-btree

  python scripts/profile-cli.py --use-cache --disable-eos --num-tokens 128     --adaptor "$adaptor" --circuits fully_factorized --n-tokens 8,16 --ranks 1     --modes mtp,speculative --output-dir outputs/results/memory_usage/${adaptor}-ff
done
