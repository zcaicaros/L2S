FROM nvidia/cuda:11.1.0-devel-ubuntu18.04
FROM python:3.8.10

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/*



