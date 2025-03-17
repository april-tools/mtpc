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
                model.sum_weight_head.encoder.n_layer=1 \
                model.token_head.encoder.n_layer=1 \
                model.token_head.expander.expander_type=linear \
                lm.model.freeze=true \
                lm.model.encoder_only=true \
                model.mt_head.freeze_unembedding=true \
                training.device_batch_size=16 \
                training.expname=basharin-apkl-n-4-r-1
