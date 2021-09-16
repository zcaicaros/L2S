FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/* \
 # create a user with home dir
 && useradd -md /home/l2s l2s \
 && chown -R l2s:l2s /home/l2s \
 && echo l2s:l2s | chpasswd

# switch to user
USER l2s
ENV PATH="/home/l2s/.local/bin:${PATH}"
