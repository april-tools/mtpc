#!/bin/bash

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
	data=tulu3-llama3-packed \
    training=tulu3-evabyte-1epoch \
	lm=llama3-2-3b-byte \
    model=mtp \
    adaptor=none \
    mt_head=linear-evabyte \
    circuit=hmm \
    circuit.n_token=8 \
    circuit.n_component=32 \
    training.device_batch_size=1 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
	data.val_bin=null \
	mt_head.hyperparameters.share_sum_weights=false \
    mt_head.hyperparameters.contextual_hmm_weights=true \
    mt_head.hyperparameters.init_hmm_identity=true \
	training.learning_rate=0.0003 \
    training.expname=llama-lr-3e-4-no-lora-hmm-n-8-r-32
