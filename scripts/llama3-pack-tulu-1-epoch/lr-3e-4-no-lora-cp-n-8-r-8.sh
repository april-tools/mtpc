# n=8, r=8, no lora, cp

torchrun --standalone \
    --nproc_per_node=$GPUS \
	-m mtp.train \
	data=tulu3-llama3-packed \
	training=tulu3-evabyte-1epoch \
	lm=llama3-2-3b-byte \
	model=mtp \
	adaptor=none \
	mt_head=linear-evabyte \
	circuit=cp \
	circuit.n_token=8 \
	circuit.n_component=8 \
	training.device_batch_size=1 \
	model.model.beta=0 \
	model.model.gamma=0.9 \
    data.val_bin=null \
    training.learning_rate=0.0003 \
    training.expname=llama-lr-3e-4-no-lora-cp-n-8-r-8
