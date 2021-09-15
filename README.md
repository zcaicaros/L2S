# Learning to Search for Job Shop Scheduling via Deep Reinforcement Learning

## Installation
python 3.8.10

CUDA 11.1

pytorch 1.9.0 with CUDA 11.1

[PyG](https://github.com/pyg-team/pytorch_geometric) 2.0.0 (with pytorch 1.9.0 and CUDA 11.1)

Matplotlib 3.4.3

[Or-Tools](https://github.com/google/or-tools) 9.0.9048

To create a docker image named `l2s_image` with all dependencies installed:
```
docker build -t l2s_image <dir-to-dockerfile>
```
To create a container named `l2s_container` with the image `l2s_image` and stay within it:
```
docker run --gpus all --name l2s_container -it l2s_image
```
To reproduce the result in the paper, first clone the whole repo:
```
git clone https://github.com/zcaicaros/L2S.git
```
Then run:
```
python3 test_learned.py
```