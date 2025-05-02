#!/bin/bash

# Experiments on 1k steps on tulu 3 using EvaByte-SFT

# n=8, r=16, lora, cp
../uv/uv run torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=tulu3 \
    training=tulu3-evabyte-full \
    lm=evabyte \
    model=mtp \
    circuit=cp \
    adaptor=lora-last-8 \
    mt_head=linear-evabyte \
    circuit.n_token=8 \
    circuit.n_component=16 \
    training.device_batch_size=2 \
    data.vocab_size=320 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
    training.expname=full-tulu-evabyte-lora-last-8-cp-n-8-r-16
