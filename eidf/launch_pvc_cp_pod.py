# Example usage.

import time

from kubejobs.jobs import KubernetesJob, KueueQueue
from rich import print
import argparse
from config import username, email, pvc_name, gpu_product, mtp_root_name

# unique id generated using time


parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str)
args = parser.parse_args()

time = 60*60

job = KubernetesJob(
    name=f"{username}-pvc-cp",
    image="nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04",
    kueue_queue_name=KueueQueue.INFORMATICS,
    command=["/bin/sh", "-c"],
    args=[f"apt -y update && apt -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get -y install wget sshfs git smbclient; cd /{mtp_root_name} ; sleep {time}"],
    gpu_type="nvidia.com/gpu",
    gpu_product=gpu_product,
    gpu_limit=1,
    # shm_size="10G",  # "200G" is the maximum value for shm_size
    backoff_limit=1,
    job_deadlineseconds=60*60,
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