#!/bin/bash

# n=8, r=1, lora, fully-factorised
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=tulu3-evabyte \
    training=tulu3-evabyte-long \
    lm=evabyte \
    model=mtp \
    circuit=fully_factorized \
    adaptor=lora-last-8 \
    mt_head=linear-evabyte \
    circuit.n_token=8 \
    training.device_batch_size=4 \
    data.vocab_size=320 \
    model.model.beta=0 \
    model.model.gamma=0.9 \
    training.expname=full-tulu-evabyte-lora-last-8-ff-n-8-r-1
