#!/bin/bash

# Code below should work with tag v0.2-kl code
# produces throughput_models.jsonl 
$MTP_ROOT/bin/compute_throughput_for_models outputs/models/finewebedu10B-full
# produces validate_models.jsonl
$MTP_ROOT/bin/validate_models outputs/models/finewebedu10B-full

# NOTE: Code below uses our cached results
# To reproduce, replace the jsonl files with those generated above
OUTFOLDER=$MTP_ROOT/scripts/basharin-ablation-fineweb-full

###### Plots ######
# Which loss contributes to better acceptance rate?
python mtp/plots/plot_accepted_tokens.py \
	--results $OUTFOLDER/throughput_models.jsonl \
	--filter-experiments basharin-cp-akl-n-4-r-2 basharin-cp-ce-n-4-r-2 basharin-cp-akl+ce-n-4-r-2

python mtp/plots/plot_metrics_compare.py \
	--metric-results $OUTFOLDER/validate_models.jsonl \
	--experiments basharin-cp-akl+ce-n-4-r-2 basharin-cp-akl-n-4-r-2 basharin-cp-ce-n-4-r-2 \
	--metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_4

# Does discounting help? Yes, seems so.
python mtp/plots/plot_accepted_tokens.py \
	--results $OUTFOLDER/throughput_models.jsonl \
	--filter-experiments basharin-cp-akl+ce-n-4-r-2 basharin-cp-akl+ce-n-4-r-2-g-80

python mtp/plots/plot_metrics_compare.py \
	--metric-results $OUTFOLDER/validate_models.jsonl \
	--experiments basharin-cp-akl+ce-n-4-r-2 basharin-cp-akl+ce-n-4-r-2-g-80 \
	--metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_4

# Does increasing the rank help?
python mtp/plots/plot_accepted_tokens.py \
	--results $OUTFOLDER/throughput_models.jsonl \
	--filter-experiments basharin-cp-akl+ce-n-4-r-1 basharin-cp-akl+ce-n-4-r-2 basharin-cp-akl+ce-n-4-r-4

python mtp/plots/plot_metrics_compare.py \
	--metric-results $OUTFOLDER/validate_models.jsonl \
	--experiments basharin-cp-akl+ce-n-4-r-1 basharin-cp-akl+ce-n-4-r-2 basharin-cp-akl+ce-n-4-r-4 \
	--metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_4

# Does including an HMM help?
python mtp/plots/plot_metrics_compare.py \
	--metric-results $OUTFOLDER/validate_models.jsonl \
	--experiments basharin-cp-akl+ce-n-4-r-4 basharin-hmm-akl+ce-n-4-r-4 \
	--metrics kl_loss_at_1 kl_loss_at_2 kl_loss_at_3 kl_loss_at_4
