FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel


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
RUN pip install --user --upgrade pip \
 && pip install --user torch-scatter -f https://data.pyg.org/whl/torch-1.7.0+cu110.html \
 && pip install --no-cache-dir torch-sparse -f https://data.pyg.org/whl/torch-1.7.0+cu110.html \
 && pip install --user torch-cluster -f https://data.pyg.org/whl/torch-1.7.0+cu110.html \
 && pip install --user torch-spline-conv -f https://data.pyg.org/whl/torch-1.7.0+cu110.html \
 && pip install --user matplotlib \
 && pip install --user ortools \
 && pip install --user --upgrade pip

