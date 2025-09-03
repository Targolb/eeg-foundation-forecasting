# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# system deps useful for audio/signal, plotting, and file IO
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git vim less htop wget curl unzip \
    libsndfile1 ffmpeg \
    libgl1 libglib2.0-0 \
    graphviz \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

WORKDIR /workspace
