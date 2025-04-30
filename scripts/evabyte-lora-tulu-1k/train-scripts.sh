#!/bin/bash

# Experiments on 1k steps on tulu 3 using EvaByte-SFT

# n=8, r=4, no lora, cp
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=cp \
	adaptor=none \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=4 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-cp-n-8-r-4

# n=8, r=2, lora, cp
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=cp \
	adaptor=lora-last-8 \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=2 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-lora-last-8-cp-n-8-r-2

# n=8, r=4, lora, cp
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=cp \
	adaptor=lora-last-8 \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=4 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-lora-last-8-cp-n-8-r-4

# n=8, r=8, lora, cp
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=cp \
	adaptor=lora-last-8 \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=8 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-lora-last-8-cp-n-8-r-8


# n=8, r=2, lora, hmm
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=hmm \
	adaptor=lora-last-8 \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=2 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-lora-last-8-cp-n-8-r-2

# n=8, r=4, lora, hmm
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=hmm \
	adaptor=lora-last-8 \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=4 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-lora-last-8-cp-n-8-r-4

# n=8, r=8, lora, hmm
torchrun --standalone \
	--nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3 \
	training=tulu3-evabyte \
	lm=evabyte \
	model=mtp \
	circuit=hmm \
	adaptor=lora-last-8 \
	mt_head=linear-evabyte \
	circuit.n_token=8 \
	circuit.n_component=8 \
	training.device_batch_size=8 \
	data.vocab_size=320 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
	training.expname=evabyte-lora-last-8-cp-n-8-r-8
