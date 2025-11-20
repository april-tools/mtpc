#!/bin/bash

# Create baseline files
for GPU in L40S RTX-3090
do
	cd $MTP_ROOT/outputs/results/$GPU
	cat throughput_evabyte_full-run-100_100.jsonl | grep 'adaptor": "none"' > throughput-raw-no-lora-100-100.jsonl
	cat throughput_evabyte_full-run-100_100.jsonl | grep 'stp-evabyte' >> throughput-raw-no-lora-100-100.jsonl

	cat throughput_evabyte_full-run-100_100.jsonl | grep 'adaptor": "lora-last-16"' > throughput-raw-lora-last-16-100-100.jsonl
	cat throughput_evabyte_full-run-100_100.jsonl | grep 'stp-evabyte' >> throughput-raw-lora-last-16-100-100.jsonl


	cat throughput_evabyte_full-run-100_100.jsonl | grep 'stp-evabyte' > throughput-sampling-no-lora.jsonl
	cat *-sampling-no-lora-*.jsonl >> throughput-sampling-no-lora.jsonl

	cat throughput_evabyte_full-run-100_100.jsonl | grep 'stp-evabyte' > throughput-sampling-lora-last-16.jsonl
	cat *-sampling-lora-last-16-*.jsonl >> throughput-sampling-lora-last-16.jsonl
done


cd $MTP_ROOT

mkdir -p outputs/tables


for GPU in L40S RTX-3090
do
	python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-evabyte-sampling-no-lora-1024-250.jsonl" --ntokens 8 --ncomponents 1 8 16 32 64 128 --decoding sampling --circuits cp --id acc-rate-$GPU-no-lora-cp-comparison --save
done


for model in "lora-last-16" "no-lora"
do
	for GPU in L40S RTX-3090
	do
		echo -e "Processing ${model}:${GPU} accepted tokens and throughput tables"
		python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --step 900 --ntokens 8 16 --ncomponents 1 32 --decoding sampling --id dummy --save > "outputs/tables/acc-rate-$GPU-$model-1024-250.txt"
		python mtp/plots/plot_throughput_speculative.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --step 900 --ntokens 1 8 16 --ncomponent 1 32 --decoding sampling --id dummy --save > "outputs/tables/throughput-$GPU-$model-1024-250.txt"

		echo -e "Processing ${model}:${GPU} raw throughput"
		# The raw throughput plots
		python mtp/plots/plot_throughput_speculative.py "outputs/results/$GPU/throughput_evabyte_full-run-100_100.jsonl"  --ntokens 1 8 16 --ncomponent 1 32  --adaptor $model --decoding sampling --id raw-throughput-$GPU-$model --save

		echo -e "Processing ${model}:${GPU} accepted token plots"
		python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --ntokens 8 --ncomponents 1 32 --decoding sampling --id acc-rate-$GPU-$model-n-8 --save
		python mtp/plots/plot_accepted_tokens.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --ntokens 16 --ncomponents 1 32 --decoding sampling --id acc-rate-$GPU-$model-n-16 --save

		echo -e "Processing ${model}:${GPU} throughput plots"
		python mtp/plots/plot_throughput_speculative.py "outputs/results/$GPU/throughput-sampling-$model.jsonl" --step 900 --ntokens 1 8 16 --ncomponent 1 32 --decoding sampling --id throughput-$GPU-$model --save > /dev/null
	done
done


# Table 1
python mtp/tables/table_1_cp_rank_comparison.py --raw-throughput-file outputs/results/L40S/tmp/throughput_evabyte_full-run-100_100.jsonl --spec-throughput-file outputs/results/L40S/throughput-evabyte-sampling-no-lora-1024-250.jsonl
python mtp/tables/table_1_cp_rank_comparison.py --raw-throughput-file outputs/results/RTX-3090/throughput_evabyte_full-run-100_100.jsonl --spec-throughput-file outputs/results/RTX-3090/throughput-evabyte-sampling-no-lora-1024-250.jsonl


python mtp/tables/table_2_throughput_longer.py --raw-throughput-file outputs/results/RTX-3090/throughput_evabyte_full-run-100_100.jsonl --spec-throughput-file outputs/results/RTX-3090/throughput-evabyte-sampling-no-lora-1024-250.jsonl
python mtp/tables/table_2_throughput_longer.py --raw-throughput-file outputs/results/L40S/throughput_evabyte_full-run-100_100.jsonl --spec-throughput-file outputs/results/L40S/throughput-evabyte-sampling-no-lora-1024-250.jsonl

python mtp/tables/table_3_throughput_lora.py --raw-throughput-file outputs/results/L40S/throughput_evabyte_full-run-100_100.jsonl --spec-throughput-file outputs/results/L40S/throughput-evabyte-sampling-lora-continued-1024-250.jsonl
