FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel


ARG project=l2s
ARG username=czhang
ARG password=czhang

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    apt-utils \
    git \
    sudo \
 && rm -rf /var/lib/apt/lists/* \
 # create a user with home dir
 && useradd -md /home/${project} ${username} \
 && chown -R ${username} /home/${project} \
 && echo ${username}:${password} | chpasswd \
 && sudo usermod -a -G sudo ${username}

# switch to user
USER ${username}
ENV PATH="/home/${project}/.local/bin:${PATH}"