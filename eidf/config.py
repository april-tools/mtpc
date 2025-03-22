# Please configure
username = "evankri"
email = "Emile.van.Krieken@ed.ac.uk"

mtp_root_name = 'mtp'

# Secrets
github_secret_name = f"{username}-git-token-2025"
wandb_secret_name = "wandb-key"
hf_secret_name = "hf-key"

# GPU setup
# Choose from: [NVIDIA-A100-SXM4-80GB, NVIDIA-A100-SXM4-40GB, NVIDIA-H100-80GB-HBM3]. For some reason, when I try H100, the job does not queue due to some memory error?... So defaulting to A100 for now. 
gpu_product = "NVIDIA-A100-SXM4-80GB" 
gpu_limit = 2
memory_limit = "50Gi"
cpus_limit = 16
pvc_size = "100Gi"

# If true, the job will be prioritized but killed after 24 hours. **Only works when gpu_limit = 1.** Useful for testing. 
short_job = False 

# Data chunks
data_chunks = 103