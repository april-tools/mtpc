#!/bin/bash


cd mtp/plots
python figure1.py
cd ../..

python mtp/plots/figure3.py --csv mtp/plots/mtpc3-evabyte-sampling-nolora.csv
python mtp/plots/figure3.py --csv mtp/plots/mtpc3-evabyte-argmax-nolora.csv
python mtp/plots/figure3.py --csv mtp/plots/mtpc3-llama-sampling-nolora.csv
python mtp/plots/figure3.py --csv mtp/plots/mtpc3-llama-argmax-nolora.csv


python mtp/plots/plot_accepted_tokens_over_window.py outputs/results/L40S/throughput-argmax-evabyte-no-lora-1024-250.jsonl outputs/results/L40S/throughput-argmax-llama-no-lora-1024-250.jsonl --ntokens 8 16 --ncomponents 1 32 --decoding argmax --circuits ff cp hmm btree --id rq2-argmax --save
python mtp/plots/plot_accepted_tokens_over_window.py outputs/results/L40S/throughput-sampling-evabyte-no-lora-1024-250.jsonl outputs/results/L40S/throughput-sampling-llama-no-lora-1024-250.jsonl --ntokens 8 16 --ncomponents 1 32 --decoding sampling --circuits ff cp hmm btree --id rq2-sampling --save

python mtp/plots/plot_accepted_tokens_over_lora.py outputs/results/L40S/throughput-argmax-evabyte-lora-continued-1024-250.jsonl outputs/results/L40S/throughput-argmax-llama-lora-continued-1024-250.jsonl --ntokens 8 --ncomponents 1 32 --decoding argmax --circuits ff btree --id rq3-argmax-n-8 --save
python mtp/plots/plot_accepted_tokens_over_lora.py outputs/results/L40S/throughput-sampling-evabyte-lora-continued-1024-250.jsonl outputs/results/L40S/throughput-sampling-llama-lora-continued-1024-250.jsonl --ntokens 8 --ncomponents 1 32 --decoding sampling --circuits ff btree --id rq3-sampling-n-8 --save

python mtp/plots/plot_accepted_tokens_over_lora.py outputs/results/L40S/throughput-argmax-evabyte-lora-continued-1024-250.jsonl outputs/results/L40S/throughput-argmax-llama-lora-continued-1024-250.jsonl --ntokens 16 --ncomponents 1 32 --decoding argmax --circuits ff btree --id rq3-argmax-n-16 --save
python mtp/plots/plot_accepted_tokens_over_lora.py outputs/results/L40S/throughput-sampling-evabyte-lora-continued-1024-250.jsonl outputs/results/L40S/throughput-sampling-llama-lora-continued-1024-250.jsonl --ntokens 16 --ncomponents 1 32 --decoding sampling --circuits ff btree --id rq3-sampling-n-16 --save



for GPU in L40S 3090RTX
do
	echo -e "%#####################################################################################"
	echo -e "%##################################  $GPU   ##########################################"
	echo -e "%#####################################################################################"
	for model in evabyte llama
	do
		MODE="sampling"
		PATH_TO_RAW="outputs/results/$GPU/throughput-${MODE}-${model}-raw-1024-100.jsonl"
		echo -e "%#####################################################################################"
		echo -e "%##################################  $model   ########################################"
		echo -e "%#####################################################################################"
		echo -e "%*************************************************************************************"
		echo -e "%############################      Sampling         ##################################"
		echo -e "%*************************************************************************************"
		## Table 1 (Sampling)

		python mtp/tables/table_1_cp_rank_comparison.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/$GPU/throughput-sampling-$model-no-lora-1024-250.jsonl

		## Table 2 (Sampling)

		python mtp/tables/table_2_throughput_longer.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/$GPU/throughput-sampling-$model-no-lora-1024-250.jsonl

		## Table 3 (Sampling)

		python mtp/tables/table_3_throughput_lora.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/$GPU/throughput-sampling-$model-lora-continued-1024-250.jsonl

		MODE="argmax"
		PATH_TO_RAW="outputs/results/$GPU/throughput-${MODE}-${model}-raw-1024-100.jsonl"
		echo -e "%#####################################################################################"
		echo -e "%*************************************************************************************"
		echo -e "%#################################   Greedy   ########################################"
		echo -e "%*************************************************************************************"

		## Table 1 (Greedy)

		python mtp/tables/table_1_cp_rank_comparison.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/$GPU/throughput-argmax-$model-no-lora-1024-250.jsonl

		## Table 2 (Greedy)

		python mtp/tables/table_2_throughput_longer.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/$GPU/throughput-argmax-$model-no-lora-1024-250.jsonl

		## Table 3 (Greedy)

		python mtp/tables/table_3_throughput_lora.py --raw-throughput-file $PATH_TO_RAW --spec-throughput-file outputs/results/$GPU/throughput-argmax-$model-lora-continued-1024-250.jsonl
	done
done
