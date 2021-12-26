# Learning to Search for Job Shop Scheduling via Deep Reinforcement Learning

## System Requirement
Ubuntu 18.04.5 LTS 

python 3.8.3

pytorch 1.7.0+cu110 (torchvision 0.8.0, torchaudio 0.7.0)
```commandline
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/cu110/torch_stable.html
```

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
pip install --user matplotlib==3.4.3
pip install --user ortools==9.0.9048
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
