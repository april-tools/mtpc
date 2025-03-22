# Example usage.

import time

from kubejobs.jobs import KubernetesJob, KueueQueue
from rich import print
import argparse
from config import username, email, github_secret_name, wandb_secret_name, hf_secret_name, gpu_product, gpu_limit, mtp_root_name, data_chunks, pvc_size, memory_limit, cpus_limit, short_job
from create_pvc import create_pvc, pvc_name

# unique id generated using time

unique_id = time.strftime("%Y%m%d%H%M%S")

# Give a name to the PVC folder (here will be the code, data and models)


install_script_name = "run_mtp_EIDF.sh"
link = f"http://files.emilevankrieken.com/{install_script_name}"

parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str)
parser.add_argument('--branch', type=str, default="eidf-setup")
parser.add_argument('--gpu_limit', type=int, default=gpu_limit)
parser.add_argument('--gpu_product', type=str, default=gpu_product)
parser.add_argument('--data_chunks', type=int, default=data_chunks)
parser.add_argument('--pvc_size', type=str, default=pvc_size)
parser.add_argument('--memory_limit', type=str, default=memory_limit)
parser.add_argument('--cpus_limit', type=int, default=cpus_limit)
parser.add_argument('--short_job', type=bool, default=short_job)
args = parser.parse_args()

cuda_vis = ",".join([f"{i}" for i in range(args.gpu_limit)])
env_vars = {
    "MTP_PVC_ROOT": f"/{mtp_root_name}",
    "MTP_DATA_CHUNKS": str(args.data_chunks),
    "MTP_GIT_BRANCH": args.branch,
    "GPU": str(args.gpu_limit),
    "CUDA_VISIBLE_DEVICES": cuda_vis,
}

script_name = args.script[:-3]
create_pvc(username, script_name, args.pvc_size)
labels = {"kueue.x-k8s.io/priority-class": "short-workload-high-priority"} if args.short_job else None

job = KubernetesJob(
    name=f"{username}-mtp-{script_name}",
    image="nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04",
    kueue_queue_name=KueueQueue.INFORMATICS,
    command=["/bin/sh", "-c"],
    args=[f"apt -y update && apt -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get -y install wget; wget {link} ; chmod +x {install_script_name} ; ./{install_script_name} {args.script} {args.branch}"],
    gpu_type="nvidia.com/gpu",
    gpu_product=args.gpu_product,
    gpu_limit=args.gpu_limit,
    cpu_request=str(args.cpus_limit),
    ram_request=args.memory_limit,
    # shm_size="10G",  # "200G" is the maximum value for shm_size
    backoff_limit=1,
    env_vars=env_vars,
    labels=labels,
    secret_env_vars={"WANDB_API_KEY": {"secret_name": wandb_secret_name, "key": "api_key"}, "HF_TOKEN": {"secret_name": hf_secret_name, "key": "api_key"}, "GIT_TOKEN": {"secret_name": github_secret_name, "key": "token"}},
    job_deadlineseconds=60*60*24 if args.short_job else 60*60*60,
    user_name=f'{username}-infk8s',
    user_email=email,
    volume_mounts={
        "mtp-pvc": {
            "pvc": pvc_name(username, script_name),
            "mountPath": f"/{mtp_root_name}"
        }
    }
)

if __name__ == "__main__":
    job_yaml = job.generate_yaml()
    print(job_yaml)
    job.run()
