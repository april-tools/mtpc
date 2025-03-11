#!/bin/sh
echo "Installing unix dependencies..."

cd /home
DEBIAN_FRONTEND=noninteractive apt-get -y install wget gpg curl python3 python-is-python3 pip git nano sudo build-essential

pip install torch
python3 -c "import torch; print(torch.cuda.get_device_name(0));print(torch.cuda.get_device_name(1))"


# echo "Installing gh..."
# wget https://github.com/cli/cli/releases/download/v2.65.0/gh_2.65.0_linux_386.tar.gz
# tar -xvf gh_2.65.0_linux_386.tar.gz
# mv gh_2.65.0_linux_386 gh




# echo "Logging in to github..."
# echo $GIT_TOKEN > token.txt
# ./gh/bin/gh auth login --with-token < token.txt
# ./gh/bin/gh auth setup-git

# echo "Cloning repository..."
# git clone https://github.com/PiotrNawrot/nanoGPT.git


# cd nanoGPT
# git checkout eidf-setup

# echo "Installing dependencies..."

# pip install --upgrade pip setuptools wheel psutil
# pip install -r requirements.txt
# python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count());print(torch.cuda.get_device_name(0));print(torch.cuda.get_device_name(1))"
# pip install flash-attn --no-build-isolation

# # Source the env variables
# chmod +x env.sh
# ./env.sh

# nvidia-smi
# echo " lol new version :D"
# python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count());print(torch.cuda.get_device_name(0));print(torch.cuda.get_device_name(1))"
# # Override some env variables
# export GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_MODE=online

# nvidia-smi

# # # Run the script
# ./scripts/basharin-ablation.sh