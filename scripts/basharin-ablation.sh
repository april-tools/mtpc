#!/bin/bash

# ##  n=2, r=2, gamma=.8
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=.8 \
#                 model.model.beta=0 \
#                 model.sum_weight_head.encoder.n_layer=0 \
#                 model.token_head.encoder.n_layer=0 \
#                 model.token_head.expander.expander_type=linear \
#                 lm.model.freeze=true \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=true \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-b-0

# ##  n=2, r=2, beta=1  - KL Only
# # Run should take ~2 days on 2 A100s
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=1 \
#                 model.model.beta=1 \
#                 model.sum_weight_head.encoder.n_layer=0 \
#                 model.token_head.encoder.n_layer=0 \
#                 model.token_head.expander.expander_type=linear \
#                 lm.model.freeze=true \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=true \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-b-1

# =========== All models below are beta=0, gamma=1 ==============
##  n=2, r=2
../uv/uv run torchrun --standalone \
                --nproc_per_node=$GPUS \
                -m mtp.train \
                data=finewebedu10B \
                training=finewebedu \
                lm=finewebedu \
                model=basharin \
                model.n_token=2 \
                model.n_component=2 \
                model.model.gamma=1 \
                model.model.beta=0 \
                model.sum_weight_head.encoder.n_layer=0 \
                model.token_head.encoder.n_layer=0 \
                model.token_head.expander.expander_type=linear \
                lm.model.freeze=true \
                lm.model.encoder_only=true \
                model.mt_head.freeze_unembedding=true \
                training.device_batch_size=16 \
                training.expname=basharin-n-2-r-2-b-0-g-1

# ##  n=2, r=2, mlp
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=1 \
#                 model.model.beta=0 \
#                 model.sum_weight_head.encoder.n_layer=0 \
#                 model.token_head.encoder.n_layer=0 \
#                 model.token_head.expander.expander_type=mlp \
#                 lm.model.freeze=true \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=true \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-mlp

# ##  n=2, r=2, sum_transformer_layer=1
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=1 \
#                 model.model.beta=0 \
#                 model.sum_weight_head.encoder.n_layer=1 \
#                 model.token_head.encoder.n_layer=0 \
#                 model.token_head.expander.expander_type=linear \
#                 lm.model.freeze=true \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=true \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-sumtrans-1

# ##  n=2, r=2, token_transformer_layer=1
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=1 \
#                 model.model.beta=0 \
#                 model.sum_weight_head.encoder.n_layer=0 \
#                 model.token_head.encoder.n_layer=1 \
#                 model.token_head.expander.expander_type=linear \
#                 lm.model.freeze=true \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=true \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-toktrans-1

# ##  n=2, r=2, unfreeze head
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=1 \
#                 model.model.beta=0 \
#                 model.sum_weight_head.encoder.n_layer=0 \
#                 model.token_head.encoder.n_layer=0 \
#                 model.token_head.expander.expander_type=linear \
#                 lm.model.freeze=true \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=false \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-unfreeze-head

# ##  n=2, r=2, unfreeze LM
# torchrun --standalone \
#                 --nproc_per_node=$GPUS \
#                 -m mtp.train \
#                 data=finewebedu10B \
#                 training=finewebedu \
#                 lm=finewebedu \
#                 model=basharin \
#                 model.n_token=2 \
#                 model.n_component=2 \
#                 model.model.gamma=1 \
#                 model.model.beta=0 \
#                 model.sum_weight_head.encoder.n_layer=0 \
#                 model.token_head.encoder.n_layer=0 \
#                 model.token_head.expander.expander_type=linear \
#                 lm.model.freeze=false \
#                 lm.model.encoder_only=true \
#                 model.mt_head.freeze_unembedding=true \
#                 training.device_batch_size=16 \
#                 training.expname=basharin-n-2-r-2-unfreeze-LM
