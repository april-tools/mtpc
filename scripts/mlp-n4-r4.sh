GPUS=2
DEVICE_BATCH_SIZE=4
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