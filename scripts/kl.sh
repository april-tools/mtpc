# Run should take ~2 days on 2 A100s
##  n=2, r=2, beta=1  - KL Only
torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=2 \
    model.n_component=2 \
    model.model.gamma=1 \
    model.model.beta=1 \
    model.sum_weight_head.encoder.n_layer=0 \
    model.token_head.encoder.n_layer=0 \
    model.token_head.expander.expander_type=linear \
    lm.model.freeze=true \
    lm.model.encoder_only=false \
    model.mt_head.freeze_unembedding=true \
    training.device_batch_size=16 \
    training.val_loss_every=200 \
    training.save_model_every=3000 \
    training.expname=basharin-n-2-r-2-b-1