#!/bin/sh
echo "Installing unix dependencies..."

cd /home
DEBIAN_FRONTEND=noninteractive apt-get -y install wget gpg curl python3 python-is-python3 pip git nano sudo build-essential

echo "Installing gh..."
wget https://github.com/cli/cli/releases/download/v2.65.0/gh_2.65.0_linux_386.tar.gz
tar -xvf gh_2.65.0_linux_386.tar.gz
mv gh_2.65.0_linux_386 gh

echo "Logging in to github..."
echo $GIT_TOKEN > token.txt
./gh/bin/gh auth login --with-token < token.txt
./gh/bin/gh auth setup-git

echo "Cloning repository..."
git clone https://github.com/PiotrNawrot/nanoGPT.git


echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="uv" sh

cd nanoGPT
git checkout eidf-setup

echo "Installing dependencies..."

../uv/uv venv --python 3.10
source .venv/bin/activate
../uv/uv pip install --upgrade pip setuptools wheel psutil
../uv/uv pip install -r requirements.txt
../uv/uv pip install flash-attn --no-build-isolation

# Source the env variables
# chmod +x env.sh
# ./env.sh

nvidia-smi
../uv/uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count());print(torch.cuda.get_device_name(0));print(torch.cuda.get_device_name(1))"
# Override some env variables
# Source this file from the root directory of the project
export MTP_ROOT=`pwd`
export GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=online

export OMP_NUM_THREADS=1
# Below allows our results to be reproducible
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Download first 10 chunks of fineweb-edu train dataset
# ./bin/download_data 10 --dataset fineweb-edu
uv run eidf/download_data.py 10 --dataset fineweb-edu

# Echo contents of requirements.txt
cat requirements.txt

# Run the script
./scripts/basharin-ablation.sh