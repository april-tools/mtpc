# Please configure
username = "s2457990"
email = "l.loconte@sms.ed.ac.uk"

mtp_root_name = 'mtp'

# Secrets
github_secret_name = f"s2457990-infk8s-git-token"
wandb_secret_name = "s2457990-infk8s-wandb-key"
hf_secret_name = "s2457990-infk8s-hf-key"

# GPU setup
# Choose from: [NVIDIA-A100-SXM4-80GB, NVIDIA-A100-SXM4-40GB, NVIDIA-H100-80GB-HBM3]. For some reason, when I try H100, the job does not queue due to some memory error?... So defaulting to A100 for now. 
gpu_product = "NVIDIA-A100-SXM4-40GB"
gpu_limit = 2
memory_limit = "64Gi"
cpus_limit = 16
pvc_size = "256Gi"

# If true, the job will be prioritized but killed after 24 hours. **Only works when gpu_limit = 1.** Useful for testing. 
short_job = False 

# Data chunks
data_chunks = 103

