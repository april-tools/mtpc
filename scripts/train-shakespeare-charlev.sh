#!/bin/bash

n_layer=6
n_head=4
n_embd=256

# Train the autoregressive LLM model
torchrun --standalone --nproc_per_node=$GPUS -m mtp.train data=shakespeare_char training=shakespeare_char \
  model=stp \
  lm.n_layer=$n_layer lm.n_head=$n_head lm.n_embd=$n_embd \
  lm.model.encoder_only=false \
  training.save_model_every=100

exit

checkpoint=  # to set

# Run the multi-token prediction models
for model in mtp-cp mtp-hmm;
do
  for n_token in 2 4 6;
  do
    for n_component in 2 4 8;
    do
      for kl in full binary_approx;
      do
        torchrun --standalone --nproc_per_node=$GPUS -m mtp.train data=shakespeare_char training=shakespeare_char \
          model=$model \
          model.n_token=$n_token \
          model.n_component=$n_component \
          model.model.kl_algorithm=$kl \
          model.mt_head_hparams.expander_type=linear \
          lm.n_layer=$n_layer lm.n_head=$n_head lm.n_embd=$n_embd \
          lm.model.encoder_only=false \
          lm.model.freeze=true \
          lm.model.lm=null lm.model.from_checkpoint=$checkpoint \
          training.save_model_every=100
      done
    done
  done
done

