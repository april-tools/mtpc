#linear vs mlp for token expander heads
#1 transformer layer vs 0 transformer layers for token expander heads and sum layer heads
#freeze output layer vs not freeze output layer
#freeze LM vs not freeze LM

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
    model.mt_head.freeze_unembedding=false \
    training.expname="debug"
