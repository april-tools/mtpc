n=4
r=1

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=$n \
    model.n_component=$r \
    training.device_batch_size=8 \
    training.val_loss_every=20 \
    training.save_model_every=100 \
    model.token_head.expander.expander_type="linear" \
    training.expname="token_haed_expander linear"

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=$n \
    model.n_component=$r \
    training.device_batch_size=8 \
    training.val_loss_every=20 \
    training.save_model_every=100 \
    model.token_head.expander.expander_type="mlp" \
    training.expname="token_haed_expander mlp"

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=$n \
    model.n_component=$r \
    training.device_batch_size=8 \
    training.val_loss_every=20 \
    training.save_model_every=100 \
    model.token_head.encoder.n_layer=1 \
    training.expname="token_head encoder 1"

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=$n \
    model.n_component=$r \
    training.device_batch_size=8 \
    training.val_loss_every=20 \
    training.save_model_every=100 \
    model.sum_weight_head.encoder.n_layer=1 \


torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=$n \
    model.n_component=$r \
    training.device_batch_size=8 \
    training.val_loss_every=20 \
    training.save_model_every=100 \
    lm.model.freeze=false \
    training.expname="lm not freeze"

torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=$n \
    model.n_component=$r \
    training.device_batch_size=8 \
    training.val_loss_every=20 \
    training.save_model_every=100 \
    model.mt_head.freeze_unembedding=false \
    training.expname="output not freeze"
