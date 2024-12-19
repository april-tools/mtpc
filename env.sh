export GPUS=1
# Make below true to allow model training even if some params do not receive gradients
# Needed for current circuit implementation because we ignore previous params
export FIND_UNUSED_PARAMS=True
