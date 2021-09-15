FROM python:3.8.10

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/*

# install cuda-toolkit
RUN mkdir /home/workdir \
 && wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run -P /home/workdir \
 && sudo sh /home/workdir/cuda_11.1.0_455.23.05_linux.run


