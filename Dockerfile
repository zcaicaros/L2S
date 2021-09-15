FROM nvidia/cuda:11.4.1-devel-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/*


