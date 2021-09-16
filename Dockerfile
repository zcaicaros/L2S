FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

# nothing to do with this image


FROM python:3.8.10


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/* \
 # create a user and add to the sudo group of the container
 && useradd -md /home/l2s -p l2s_passward l2s \
 # && echo l2s_passward | passwd -S l2s \
 && sudo adduser l2s sudo
 # switch to user
USER l2s



# install dependencies
RUN echo l2s_passward | sudo -S pip install \
    --upgrade pip \
    torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-geometric==1.7.2 \
    torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    matplotlib \
    ortools \
    --upgrade pip

