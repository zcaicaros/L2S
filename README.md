# Learning to Search for Job Shop Scheduling via Deep Reinforcement Learning

## Installation
python 3.8.3

CUDA 11.0

pytorch 1.7.0 with CUDA 11.0

[PyG](https://github.com/pyg-team/pytorch_geometric) 1.7.2

Matplotlib 3.4.3

[Or-Tools](https://github.com/google/or-tools) 9.0.9048

To create a docker image named `l2s_image` with `python 3.8.3`, `torch 1.7.0`,  and `CUDA` installed:
```
docker build --rm -t l2s_image <dir-to-dockerfile>
```
To create a container named `l2s_container` with the image `l2s_image` and stay within it:
```
docker run --gpus all --name l2s_container -it l2s_image bash
```
Then install dependencies:
```
cd ~
pip install --user --upgrade pip
pip install --user torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --user torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --user torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --user torch-spline-conv=1.2.1 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --user torch-geometric==1.7.2
pip install --user matplotlib
pip install --user ortools
```

## Reproducing
To reproduce the result in the paper, first clone the whole repo:
```
cd ~
git clone https://github.com/zcaicaros/L2S.git
```
Then run:
```
python3 test_learned.py
```
