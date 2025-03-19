n=4
r=1

#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=$n \
#    model.n_component=$r \
#    training.device_batch_size=8 \
#    training.val_loss_every=200 \
#    training.save_model_every=3000 \
#    model.token_head.expander.expander_type="linear" \
#    training.expname="token_haed_expander linear"
#
#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=$n \
#    model.n_component=$r \
#    training.device_batch_size=8 \
#    training.val_loss_every=200 \
#    training.save_model_every=3000 \
#    model.token_head.expander.expander_type="mlp" \
#    training.expname="token_haed_expander mlp"
#
#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=$n \
#    model.n_component=$r \
#    training.device_batch_size=8 \
#    training.val_loss_every=200 \
#    training.save_model_every=3000 \
#    model.token_head.encoder.n_layer=1 \
#    training.expname="token_head encoder 1"
#
#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=$n \
#    model.n_component=$r \
#    training.device_batch_size=8 \
#    training.val_loss_every=200 \
#    training.save_model_every=3000 \
#    model.sum_weight_head.encoder.n_layer=1 \
#    training.expname="sum_weight_head encoder 1"
#
#
#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=$n \
#    model.n_component=$r \
#    training.device_batch_size=8 \
#    training.val_loss_every=200 \
#    training.save_model_every=3000 \
#    lm.model.freeze=false \
#    training.expname="lm not freeze"

#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=$n \
#    model.n_component=$r \
#    training.device_batch_size=8 \
#    training.val_loss_every=200 \
#    training.save_model_every=3000 \
#    model.mt_head.freeze_unembedding=false \
#    training.expname="output not freeze"


torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=1 \
    model.gamma=.8 \
    model.beta=.9 \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    lm.model.freeze=true \
    lm.model.encoder_only=false \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    training.device_batch_size=2 \
    training.val_loss_every=200 \
    training.save_model_every=3000 \
    training.expname=n-4-r-1-nth-0-nts-0-linear



torchrun --standalone \
    --nproc_per_node=$GPUS \
    -m mtp.train \
    data=finewebedu10B \
    training=finewebedu \
    lm=finewebedu \
    model=basharin \
    model.n_token=4 \
    model.n_component=2 \
    model.gamma=.8 \
    model.beta=.9 \
    model.mt_head_hparams.sum_transformer_n_layer=0 \
    model.mt_head_hparams.tok_transformer_n_layer=0 \
    model.mt_head_hparams.expander_type=linear \
    lm.model.freeze=true \
    lm.model.encoder_only=false \
    model.mt_head_hparams.freeze_vocab_unembedding=true \
    training.device_batch_size=2 \
    training.val_loss_every=200 \
    training.save_model_every=3000 \
    training.expname=n-4-r-2-nth-0-nts-0-linear

#torchrun --standalone \
#    --nproc_per_node=$GPUS \
#    -m mtp.train \
#    data=finewebedu10B \
#    training=finewebedu \
#    lm=finewebedu \
#    model=basharin \
#    model.n_token=4 \
#    model.n_component=4 \
#    model.gamma=.8 \
#    model.beta=.9 \
#    model.sum_weight_head.encoder.n_layer=0 \
#    model.token_head.encoder.n_layer=0 \
#    model.token_head.expander.expander_type=linear \
#    lm.model.freeze=true \
#    lm.model.encoder_only=false \
#    model.mt_head.freeze_unembedding=true \
#    training.device_batch_size=1 \
#    training.expname="debug n-4-r-4-nth-0-nts-0-linear"
