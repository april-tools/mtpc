#!/bin/sh
echo "Installing unix dependencies..."

cd /home
DEBIAN_FRONTEND=noninteractive apt-get -y install wget gpg curl python3 python-is-python3 pip git nano sudo build-essential

cd $MTP_PVC_ROOT

if [ ! -d "$MTP_PVC_ROOT/gh" ]; then
	echo "Installing gh..."
	wget https://github.com/cli/cli/releases/download/v2.65.0/gh_2.65.0_linux_386.tar.gz
	tar -xvf gh_2.65.0_linux_386.tar.gz
	mv gh_2.65.0_linux_386 gh
fi

echo "Logging in to github..."
echo $GIT_TOKEN > token.txt
./gh/bin/gh auth login --with-token < token.txt
./gh/bin/gh auth setup-git
rm token.txt

if [ ! -d "$MTP_PVC_ROOT/uv" ]; then
	echo "Installing uv..."
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="uv" sh
fi


# On first install
if [ ! -d "nanoGPT" ]; then
	echo "Cloning repository..."
	git clone https://github.com/PiotrNawrot/nanoGPT.git
	cd nanoGPT
	git checkout $MTP_GIT_BRANCH
	../uv/uv venv --python 3.10
else
	# Else just ensure up to date
	echo "Already cloned."
	cd nanoGPT
	git status
	git fetch --all
	# This is needed in very specific cases. 
	git reset --hard origin/$MTP_GIT_BRANCH
fi
source .venv/bin/activate
echo "Installing dependencies..."
../uv/uv pip install --upgrade pip setuptools wheel psutil
../uv/uv pip install -r requirements.txt
../uv/uv pip install flash-attn --no-build-isolation

export MTP_ROOT=`pwd`
export WANDB_MODE=online

echo "GPU setup:"
echo $GPU
echo $CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS=1
# Below allows our results to be reproducible
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER=Emile

nvidia-smi
../uv/uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count());print(torch.cuda.get_device_name(0));print(torch.cuda.get_device_name(1))"


echo "Main directory:"
echo $MTP_PVC_ROOT

# Download first 10 chunks of fineweb-edu train dataset
../uv/uv run eidf/download_data.py $MTP_DATA_CHUNKS --dataset fineweb-edu

# Run the script
chmod +x scripts/"$1"
./scripts/"$1"
