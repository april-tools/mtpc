# Source this file from the root directory of the project
export MTP_ROOT=`pwd`
# Adapt below based on your gpu config
export GPUS=1
export CUDA_VISIBLE_DEVICES=0
# export GPUS=3
# export CUDA_VISIBLE_DEVICES=0,1,2
export WANDB_MODE=disabled
# export WANDB_MODE=online
export OMP_NUM_THREADS=1
# Below allows our results to be reproducible
export CUBLAS_WORKSPACE_CONFIG=:4096:8
