#!/bin/bash

# TODO: Figure out device_batch_size for 4 A100s
DEVICE_BATCH_SIZE=4


##  n=4, r=1
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin-lora \
    model.n_token=4 \
    model.n_component=1 \
    model.circuit.kind=cp \
    model.model.gamma=.8 \
    model.model.beta=0 \
    lm.model.encoder_only=true \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
	model.adaptor_params.r=32 \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-ce-lora-32-n-4-r-1 \
    compile=false


##  n=4, r=2
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin-lora \
    model.n_token=4 \
    model.n_component=2 \
    model.circuit.kind=cp \
    model.model.gamma=.8 \
    model.model.beta=0 \
    lm.model.encoder_only=true \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
	model.adaptor_params.r=32 \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-ce-lora-32-n-4-r-2 \
    compile=false


##  n=4, r=4
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin-lora \
    model.n_token=4 \
    model.n_component=4 \
    model.circuit.kind=cp \
    model.model.gamma=.8 \
    model.model.beta=0 \
    lm.model.encoder_only=true \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
	model.adaptor_params.r=32 \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-ce-lora-32-n-4-r-4 \
    compile=false


##  n=4, r=4, hmm
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin-lora \
    model.n_token=4 \
    model.n_component=4 \
    model.circuit.kind=hmm \
    model.model.gamma=.8 \
    model.model.beta=0 \
    lm.model.encoder_only=true \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
	model.adaptor_params.r=32 \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-hmm-ce-lora-32-n-4-r-4 \
    compile=false
