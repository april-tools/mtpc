# Example usage.

import time

from kubejobs.jobs import KubernetesJob, create_pvc, KueueQueue
from rich import print
import argparse

# unique id generated using time

unique_id = time.strftime("%Y%m%d%H%M%S")

# # create some persistent storage to keep around model weights, and perhaps data if you need it
# create_pvc(
#     pvc_name="tutorial-pvc", storage="10Gi", access_modes="ReadWriteOnce"
# )

env_vars = {
    "DATASET_DIR": "/data/",
    "MODEL_DIR": "/data/model/",
}

install_script_name = "run_mtp_EIDF.sh"
link = f"http://files.emilevankrieken.com/{install_script_name}"

run_script_name = "basharin-ablation.sh"


job = KubernetesJob(
    name=f"evankri-mtp-basharin-ablation",
    image="nvcr.io/nvidia/pytorch:23.10-py3",
    kueue_queue_name=KueueQueue.INFORMATICS,
    command=["/bin/sh", "-c"],
    args=[f"wget {link} ; chmod +x {install_script_name} ; ./{install_script_name}"],
    gpu_type="nvidia.com/gpu",
    # gpu_product="NVIDIA-A100-SXM4-40GB",
    gpu_limit=1,
    # shm_size="10G",  # "200G" is the maximum value for shm_size
    backoff_limit=4,
    cpu_request=24,
    ram_request=f"20G",
    env_vars=env_vars,
    secret_env_vars={"WANDB_API_KEY": {"secret_name": "wandb-key", "key": "api_key"}, "HF_TOKEN": {"secret_name": "hf-key", "key": "api_key"}, "GIT_TOKEN": {"secret_name": "evankri-git-token-2025", "key": "token"}},
    job_deadlineseconds=60*60,
    user_name='evankri-infk8s',
    user_email='Emile.van.Krieken@ed.ac.uk'
)

job_yaml = job.generate_yaml()
print(job_yaml)
job.run()