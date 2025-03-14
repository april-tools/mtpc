# Example usage.

import time

from kubejobs.jobs import KubernetesJob, KueueQueue
from rich import print
import argparse
from config import username, email, pvc_name, github_secret_name, wandb_secret_name, hf_secret_name, gpu_product, gpu_limit, mtp_root_name

# unique id generated using time

unique_id = time.strftime("%Y%m%d%H%M%S")

# Give a name to the PVC folder (here will be the code, data and models)
env_vars = {
    "MTP_PVC_ROOT": f"/{mtp_root_name}",
}

install_script_name = "run_mtp_EIDF.sh"
link = f"http://files.emilevankrieken.com/{install_script_name}"

parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str)
args = parser.parse_args()


job = KubernetesJob(
    name=f"{username}-mtp-{args.script[:-3]}",
    image="nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04",
    kueue_queue_name=KueueQueue.INFORMATICS,
    command=["/bin/sh", "-c"],
    args=[f"apt -y update && apt -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get -y install wget; wget {link} ; chmod +x {install_script_name} ; ./{install_script_name} {args.script}"],
    gpu_type="nvidia.com/gpu",
    gpu_product=gpu_product,
    gpu_limit=gpu_limit,
    # shm_size="10G",  # "200G" is the maximum value for shm_size
    backoff_limit=1,
    env_vars=env_vars,
    secret_env_vars={"WANDB_API_KEY": {"secret_name": wandb_secret_name, "key": "api_key"}, "HF_TOKEN": {"secret_name": hf_secret_name, "key": "api_key"}, "GIT_TOKEN": {"secret_name": github_secret_name, "key": "token"}},
    job_deadlineseconds=60*60*60,
    user_name=f'{username}-infk8s',
    user_email=email,
    volume_mounts={
        "mtp-pvc": {
            "pvc": f"{username}-{pvc_name}",
            "mountPath": f"/{mtp_root_name}"
        }
    }
)

if __name__ == "__main__":
    job_yaml = job.generate_yaml()
    print(job_yaml)
    job.run()