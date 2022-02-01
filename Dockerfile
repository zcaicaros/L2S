FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel


ARG project=l2s
ARG username=czhang
ARG password=czhang

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/* \
 # create a user with home dir
 && useradd -md /home/${username} ${username} \
 && chown -R ${username}:${username} /home/${username} \
 && echo ${username}:${password} | chpasswd

# switch to user
USER ${username}
ENV PATH="/home/${project}/.local/bin:${PATH}"