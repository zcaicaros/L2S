FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# nothing to do with this image


FROM python:3.8.10


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/* \
 # create a user and add to the sudo group of the container
 && useradd -md /home/l2s l2s \
 && chown -R l2s:l2s /home/l2s \
 && echo l2s:l2s | chpasswd

# switch to user
USER l2s
ENV PATH="/home/l2s/.local/bin:${PATH}"


# install dependencies
RUN pip install --user \
    --upgrade pip \
    torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html \
    torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-geometric==1.7.2 \
    torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    matplotlib \
    ortools \
    --upgrade pip

