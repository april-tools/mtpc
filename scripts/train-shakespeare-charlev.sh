#!/bin/bash

export USER=loreloc

n_layer=6
n_head=4
n_embd=256

# Train the autoregressive LLM model
#torchrun --standalone --nproc_per_node=$GPU -m mtp.train data=shakespeare_char training=shakespeare_char \
#  model=stp \
#  lm.n_layer=$n_layer lm.n_head=$n_head lm.n_embd=$n_embd \
#  lm.model.encoder_only=false \
#  training.save_model_every=100
#  training.expname=stp-shcharlev

checkpoint="logs/2025-03-22/18-33-59/model@600.pt"  # to set

# Run the multi-token prediction models
for model in mtp-cp mtp-hmm;
do
  for n_token in 4 6 8;
  do
    for n_component in 4;
    do
      for kl in full;
      do
        torchrun --standalone --nproc_per_node=$GPU -m mtp.train data=shakespeare_char training=shakespeare_char \
          model=$model \
          model.n_token=$n_token \
          model.n_component=$n_component \
          model.model.kl_algorithm=$kl \
          model.mt_head_hparams.tok_transformer_n_layer=1 \
          model.mt_head_hparams.sum_transformer_n_layer=1 \
          model.mt_head_hparams.expander_type=linear \
          lm.n_layer=$n_layer lm.n_head=$n_head lm.n_embd=$n_embd \
          lm.model.encoder_only=false \
          lm.model.freeze=true \
          lm.model.lm=null lm.model.from_checkpoint="$checkpoint" \
          training.save_model_every=100 \
          training.expname=shcharlev-transf-$model-n-$n_token-r-$n_component-kl-$kl-g0.9
      done
    done
  done
done

