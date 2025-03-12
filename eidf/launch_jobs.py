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

parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str)
args = parser.parse_args()


job = KubernetesJob(
    name=f"evankri-mtp-{args.script}",
    #image="nvcr.io/nvidia/pytorch:25.02-py3",
    image="nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04",
    kueue_queue_name=KueueQueue.INFORMATICS,
    command=["/bin/sh", "-c"],
    args=[f"apt -y update && apt -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get -y install wget; wget {link} ; chmod +x {install_script_name} ; ./{install_script_name} {args.script}"],
    gpu_type="nvidia.com/gpu",
    gpu_product="NVIDIA-H100-80GB-HBM3",
    gpu_limit=2,
    # shm_size="10G",  # "200G" is the maximum value for shm_size
    backoff_limit=1,
    env_vars=env_vars,
    secret_env_vars={"WANDB_API_KEY": {"secret_name": "wandb-key", "key": "api_key"}, "HF_TOKEN": {"secret_name": "hf-key", "key": "api_key"}, "GIT_TOKEN": {"secret_name": "evankri-git-token-2025", "key": "token"}},
    job_deadlineseconds=60*60,
    user_name='evankri-infk8s',
    user_email='Emile.van.Krieken@ed.ac.uk'
)

job_yaml = job.generate_yaml()
print(job_yaml)
job.run()