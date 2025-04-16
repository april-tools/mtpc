#!/bin/bash

##  n=4, r=4
# Iterate over different values for n_token and n_component
for n in 1 2 4 8 16 32
do
    echo "Running with batch size $n"
    
    ../uv/uv run torchrun --standalone \
		--nproc_per_node=$GPUS \
		-m mtp.train \
		data=finewebedu10B \
		training=finewebedu \
		lm=finewebedu \
		model=basharin \
		model.n_token=4 \
		model.n_component=4 \
		model.model.gamma=1 \
		model.model.kl_algorithm=binary_approx \
		model.model.beta=.9 \
		lm.model.encoder_only=false \
		model.mt_head_hparams.sum_transformer_n_layer=0 \
		model.mt_head_hparams.tok_transformer_n_layer=0 \
		model.mt_head_hparams.expander_type=linear \
		lm.model.freeze=true \
		model.mt_head_hparams.freeze_vocab_unembedding=true \
		training.device_batch_size=$n \
		model.circuit.kind=cp \
		training.num_iterations=10 \
		training.expname=basharin-akl+ce-n-4-r-4-b-$n
done
