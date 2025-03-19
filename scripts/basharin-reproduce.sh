#!/bin/bash

#  n=2, r=1
torchrun --standalone \
                --nproc_per_node=$GPUS \
                -m mtp.train \
                data=finewebedu10B \
                training=finewebedu \
                lm=finewebedu \
                model=basharin \
                model.n_token=2 \
                model.n_component=1 \
                model.model.gamma=.8 \
                model.model.beta=.9 \
                model.mt_head_hparams.sum_transformer_n_layer=0 \
                model.mt_head_hparams.tok_transformer_n_layer=0 \
                model.mt_head_hparams.expander_type=linear \
                lm.model.freeze=true \
                lm.model.encoder_only=false \
                model.mt_head_hparams.freeze_vocab_unembedding=true \
                training.device_batch_size=16 \
                training.expname=basharin-n-2-r-1

##  n=2, r=2
torchrun --standalone \
                --nproc_per_node=$GPUS \
                -m mtp.train \
                data=finewebedu10B \
                training=finewebedu \
                lm=finewebedu \
                model=basharin \
                model.n_token=2 \
                model.n_component=2 \
                model.model.gamma=.8 \
                model.model.beta=.9 \
                model.mt_head_hparams.sum_transformer_n_layer=0 \
                model.mt_head_hparams.tok_transformer_n_layer=0 \
                model.mt_head_hparams.expander_type=linear \
                lm.model.freeze=true \
                lm.model.encoder_only=false \
                model.mt_head_hparams.freeze_vocab_unembedding=true \
                training.device_batch_size=16 \
                training.expname=basharin-n-2-r-2

##  n=2, r=4
torchrun --standalone \
                --nproc_per_node=$GPUS \
                -m mtp.train \
                data=finewebedu10B \
                training=finewebedu \
                lm=finewebedu \
                model=basharin \
                model.n_token=2 \
                model.n_component=4 \
                model.model.gamma=.8 \
                model.model.beta=.9 \
                model.mt_head_hparams.sum_transformer_n_layer=0 \
                model.mt_head_hparams.tok_transformer_n_layer=0 \
                model.mt_head_hparams.expander_type=linear \
                lm.model.freeze=true \
                lm.model.encoder_only=false \
                model.mt_head_hparams.freeze_vocab_unembedding=true \
                training.device_batch_size=16 \
                training.expname=basharin-n-2-r-4
