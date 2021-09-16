FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

# nothing to do with this image


FROM python:3.8.10


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/*


RUN useradd -md /home/l2s l2s
 # && chown -R l2s:l2s /home/l2s
USER l2s_user



# install dependencies
RUN pip install \
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

