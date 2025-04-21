#!/bin/bash

export USER=loreloc


torchrun --standalone --nproc_per_node=$GPU -m mtp.train \
    data=mnistbyte training=mnistbyte \
    model=mtp \
    model.beta=0 \
    model.gamma=0.9 \
    model.model.init_from_lm_head=False \
    circuit=cp \
    circuit.n_token=16 \
    circuit.n_component=4 \
    mt_head=transformer-evabyte \
    lm=evabyte \
    adaptor=lora \
    training.learning_rate=0.001 \
    training.expname=evabyte-transf-mtp-cp-n-16-r-4-dsc-ih-lora \
    compile=False


torchrun --standalone --nproc_per_node=$GPU -m mtp.train \
    data=mnistbyte training=mnistbyte \
    model=mtp \
    model.beta=0 \
    model.gamma=0.9 \
    model.model.init_from_lm_head=False \
    circuit=hmm \
    circuit.n_token=16 \
    circuit.n_component=4 \
    mt_head=transformer-evabyte \
    lm=evabyte \
    adaptor=lora \
    training.expname=evabyte-transf-mtp-hmm-n-16-r-4-dsc-ih-lora \
    training.learning_rate=0.001 \
    compile=False


torchrun --standalone --nproc_per_node=$GPU -m mtp.train \
    data=mnistbyte training=mnistbyte \
    model=mtp \
    model.beta=0 \
    model.gamma=0.9 \
    model.model.init_from_lm_head=True \
    circuit=cp \
    circuit.n_token=16 \
    circuit.n_component=4 \
    mt_head=linear-evabyte \
    lm=evabyte \
    adaptor=lora \
    training.expname=evabyte-linear-mtp-cp-n-16-r-4-dsc-ih-lora \
    training.learning_rate=0.001 \
    compile=True


torchrun --standalone --nproc_per_node=$GPU -m mtp.train \
    data=mnistbyte training=mnistbyte \
    model=mtp \
    model.beta=0 \
    model.gamma=0.9 \
    model.model.init_from_lm_head=True \
    circuit=hmm \
    circuit.n_token=16 \
    circuit.n_component=4 \
    mt_head=linear-evabyte \
    lm=evabyte \
    adaptor=lora \
    training.expname=evabyte-linear-mtp-hmm-n-16-r-4-dsc-ih-lora \
    training.learning_rate=0.001 \
    compile=True
