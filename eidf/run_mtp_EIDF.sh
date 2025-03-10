#!/bin/sh
echo "Installing unix dependencies..."

cd /home
apt-get install wget gpg curl

apt-get install git

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
chmod +x env.sh
./env.sh

# Override some env variables
export GPUS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online

# Run the script
./scripts/basharin-ablation.sh