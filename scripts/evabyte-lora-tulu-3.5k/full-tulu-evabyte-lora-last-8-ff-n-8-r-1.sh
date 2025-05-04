#!/bin/bash

# Experiments on 3.5k steps on tulu 3 using EvaByte-SFT

# n=8, r=1, lora, fully-factorised
../uv/uv run torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=tulu3-evabyte \
    training=tulu3-evabyte-full \
    lm=evabyte \
    model=mtp \
    circuit=fully-factorised \
    adaptor=lora-last-8 \
    mt_head=linear-evabyte \
    circuit.n_token=8 \
    training.device_batch_size=8 \
    data.vocab_size=320 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
    training.expname=full-tulu-evabyte-lora-last-8-ff-n-8-r-1
