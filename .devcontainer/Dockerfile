# CUDA 11.8 + cuDNN 8 devel image (Ubuntu 22.04)
# NOTE: The Makefile's default ARCH is compute_30~52 (Kepler/Maxwell).
# CUDA 11.x only supports compute_35 and above. When using modern GPUs
# (Ampere/Ada, etc.), update ARCH in the Makefile to the matching
# architecture (e.g., compute_86, compute_89).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        pkg-config \
        git \
        wget \
        curl \
        ca-certificates \
        vim \
        sudo \
        unzip \
        libopencv-dev \
        python3 \
        python3-pip \
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user (UID/GID 1000)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda

USER ${USERNAME}
WORKDIR /workspaces/darknet
