#!/bin/bash

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    load_mtp_head_from_model=outputs/models/llama/no-lora/llama-lr-3e-4-no-lora-btree-n-8-r-32-s-1/model@900.pt \
	data=tulu3-llama3-packed \
    training=tulu3-evabyte-1epoch \
	lm=llama3-2-3b-byte \
    model=mtp \
    adaptor=lora-last-4 \
    mt_head=linear-evabyte \
    circuit=btree \
    circuit.n_token=8 \
    circuit.n_component=32 \
    circuit.n_repetition=1 \
    training.device_batch_size=1 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
    data.val_bin=null \
    training.learning_rate=0.0003 \
    training.expname=llama-lr-3e-4-lora-last-4-cont-btree-n-8-r-32-s-1
