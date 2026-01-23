#!/bin/bash


# TODO: Revisit below and see if needed

# # Create baseline files
# for GPU in L40S RTX-3090
# do
# 	cd $MTP_ROOT/outputs/results/$GPU
# 	cat throughput_evabyte_full-run-100_100.jsonl | grep 'adaptor": "none"' > throughput-raw-no-lora-100-100.jsonl
# 	cat throughput_evabyte_full-run-100_100.jsonl | grep 'stp-evabyte' >> throughput-raw-no-lora-100-100.jsonl
#
# 	cat throughput_evabyte_full-run-100_100.jsonl | grep 'stp-evabyte' > throughput-sampling-no-lora.jsonl
# 	cat *-sampling-no-lora-*.jsonl >> throughput-sampling-no-lora.jsonl
#
# done
#
#
# cd $MTP_ROOT
#
# mkdir -p outputs/tables
#
#
# for GPU in L40S RTX-3090
# do
# 	python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-evabyte-sampling-no-lora-1024-250.jsonl" --ntokens 8 --ncomponents 1 8 16 32 64 128 --decoding sampling --circuits cp --id acc-rate-$GPU-no-lora-cp-comparison --save
# done
#
#
# for model in "lora-continued" "no-lora"
# do
# 	for GPU in L40S RTX-3090
# 	do
# 		echo -e "Processing ${model}:${GPU} accepted tokens and throughput tables"
# 		python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --step 900 --ntokens 8 16 --ncomponents 1 32 --decoding sampling --id dummy --save > "outputs/tables/acc-rate-$GPU-$model-1024-250.txt"
# 		python mtp/plots/plot_throughput_speculative.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --step 900 --ntokens 1 8 16 --ncomponent 1 32 --decoding sampling --id dummy --save > "outputs/tables/throughput-$GPU-$model-1024-250.txt"
#
# 		echo -e "Processing ${model}:${GPU} raw throughput"
# 		# The raw throughput plots
# 		python mtp/plots/plot_throughput_speculative.py "outputs/results/$GPU/throughput_evabyte_full-run-100_100.jsonl"  --ntokens 1 8 16 --ncomponent 1 32  --adaptor $model --decoding sampling --id raw-throughput-$GPU-$model --save
#
# 		echo -e "Processing ${model}:${GPU} accepted token plots"
# 		python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --ntokens 8 --ncomponents 1 32 --decoding sampling --id acc-rate-$GPU-$model-n-8 --save
# 		python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --ntokens 16 --ncomponents 1 32 --decoding sampling --id acc-rate-$GPU-$model-n-16 --save
#
# 		echo -e "Processing ${model}:${GPU} throughput plots"
# 		python mtp/plots/plot_throughput_speculative.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --step 900 --ntokens 1 8 16 --ncomponent 1 32 --decoding sampling --id throughput-$GPU-$model --save > /dev/null
# 	done
# done


### Final scripts: TODO: re-run STP and update PATH_TO_RAW paths

## Evabyte


for model in evabyte llama
do
	PATH_TO_RAW="outputs/results/L40S/throughput-${model}-raw-1024-10.jsonl"
	echo -e "#####################################################################################"
	echo -e "##################################  $model   ########################################"
	echo -e "#####################################################################################"
	echo -e "*************************************************************************************"
	echo -e "############################      Sampling         ##################################"
	echo -e "*************************************************************************************"
	## Table 1 (Sampling)

	python mtp/tables/table_1_cp_rank_comparison.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/L40S/throughput-sampling-$model-no-lora-1024-250.jsonl

	## Table 2 (Sampling)

	python mtp/tables/table_2_throughput_longer.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/L40S/throughput-sampling-$model-no-lora-1024-250.jsonl

	## Table 3 (Sampling)

	python mtp/tables/table_3_throughput_lora.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/L40S/throughput-sampling-$model-lora-continued-1024-250.jsonl

	echo -e "#####################################################################################"
	echo -e "*************************************************************************************"
	echo -e "#################################   Greedy   ########################################"
	echo -e "*************************************************************************************"

	## Table 1 (Greedy)

	python mtp/tables/table_1_cp_rank_comparison.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/L40S/throughput-argmax-$model-no-lora-1024-250.jsonl

	## Table 2 (Greedy)

	python mtp/tables/table_2_throughput_longer.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/L40S/throughput-argmax-$model-no-lora-1024-250.jsonl

	## Table 3 (Greedy)

	python mtp/tables/table_3_throughput_lora.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/L40S/throughput-argmax-$model-lora-continued-1024-250.jsonl
done
