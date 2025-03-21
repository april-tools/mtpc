# Please configure
username = "evankri"
email = "Emile.van.Krieken@ed.ac.uk"

mtp_root_name = 'mtp'

# Secrets
github_secret_name = f"{username}-git-token-2025"
wandb_secret_name = "wandb-key"
hf_secret_name = "hf-key"

# GPU setup
gpu_product = "NVIDIA-H100-80GB-HBM3"
gpu_limit = 2
memory_limit = "50Gi"
cpus_limit = 16
pvc_size = "100Gi"

# Data chunks
data_chunks = 103