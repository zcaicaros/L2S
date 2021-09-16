FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

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
    torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-geometric==1.7.2 \
    torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html \
    matplotlib \
    ortools \
    --upgrade pip

