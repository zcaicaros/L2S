FROM nvidia/cuda:11.4.1-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update \
 && sudo apt-get install software-properties-common \
 && sudo add-apt-repository ppa:deadsnakes/ppa\
 && sudo apt-get update \
 && sudo apt-get -y install python3.8.10


