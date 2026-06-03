FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    MTP_ROOT=/workspace/mtpc \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/mtpc

COPY requirements.txt ./requirements.txt

RUN python3.10 -m pip install --upgrade pip setuptools wheel psutil && \
    python3.10 -m pip install -r requirements.txt && \
    python3.10 -m pip install flash-attn==2.8.3 --no-build-isolation

COPY . .

ENV PYTHONPATH=/workspace/mtpc

CMD ["bash"]
