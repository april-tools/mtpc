#!/bin/bash

##  n=2, r=2, token_transformer_layer=1
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=1 \
    model.model.gamma=1 \
    model.model.beta=0 \
    model.mt_head_hparams.sum_transformer_n_layer=1 \
    model.mt_head_hparams.tok_transformer_n_layer=1 \
    model.mt_head_hparams.expander_type=linear \
    lm.model.freeze=true \
    lm.model.encoder_only=true \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    training.device_batch_size=16 \
    training.expname=basharin-apkl-n-4-r-1
