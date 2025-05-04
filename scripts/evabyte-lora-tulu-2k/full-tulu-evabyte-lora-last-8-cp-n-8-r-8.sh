#!/bin/bash

# n=8, r=8, lora, cp
../uv/uv run torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=tulu3-evabyte \
    training=tulu3-evabyte-long \
    lm=evabyte \
    model=mtp \
    circuit=cp \
    adaptor=lora-last-8 \
    mt_head=linear-evabyte \
    circuit.n_token=8 \
    circuit.n_component=8 \
    training.device_batch_size=4 \
    data.vocab_size=320 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
    training.expname=full-tulu-evabyte-lora-last-8-cp-n-8-r-8
