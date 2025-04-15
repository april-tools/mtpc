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
git clone https://github.com/HEmile/expressive-nesy.git


echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="uv" sh

# cd uv
# ls
# cd ..

cd expressive-nesy
git checkout eidf

echo "Syncing dependencies..."
../uv/uv sync

# Check if a sweep ID is provided as a command-line argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide a sweep ID as a command-line argument."
    exit 1
fi

# Store the sweep ID from the command-line argument
SWEEP_ID="$1"

echo "Downloading data..."
cd expressive/experiments/path_planning
./download.sh
cd data

echo "Merging data..."
../../../../../uv/uv run merge.py

# Couldn't figure out how to get uv to path, so just relative paths lol. Careful with the cd's. 
echo "Running path planning experiment..."
cd ..
../../../../uv/uv run wandb agent hemile/nesy-diffusion-wc/$SWEEP_ID --count "$2"
