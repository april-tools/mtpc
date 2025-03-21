#!/bin/bash

# TODO: Figure out device_batch_size for 4 A100s
DEVICE_BATCH_SIZE=16


##  n=4, r=1
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=1 \
	model.circuit.kind=cp \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=.9 \
    lm.model.encoder_only=false \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-akl+ce-n-4-r-1


##  n=4, r=2
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=2 \
	model.circuit.kind=cp \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=.9 \
    lm.model.encoder_only=false \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-akl+ce-n-4-r-2


##  n=4, r=4
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=4 \
	model.circuit.kind=cp \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=.9 \
    lm.model.encoder_only=false \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-akl+ce-n-4-r-4


##  n=4, r=4, mlp
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=4 \
	model.circuit.kind=cp \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=.9 \
    lm.model.encoder_only=false \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=mlp \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-akl+ce-mlp-n-4-r-4


##  n=4, r=4, only akl
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=4 \
	model.circuit.kind=cp \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=1 \
    lm.model.encoder_only=false \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-akl-n-4-r-4


##  n=4, r=4, only ce
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
	model.circuit.kind=cp \
    model.n_token=4 \
    model.n_component=4 \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=0 \
    lm.model.encoder_only=true \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-cp-ce-n-4-r-4


##  n=4, r=4, hmm
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=4 \
	model.circuit.kind=hmm \
    model.model.gamma=1 \
    model.model.kl_algorithm=binary_approx \
    model.model.beta=.9 \
    lm.model.encoder_only=false \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    lm.model.freeze=true \
    training.device_batch_size=$DEVICE_BATCH_SIZE \
    training.expname=basharin-hmm-akl+ce-n-4-r-4
