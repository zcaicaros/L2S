# Learning to Search for Job Shop Scheduling via Deep Reinforcement Learning

## System Requirement
Ubuntu 18.04.5 LTS 

python 3.8.1

pytorch 1.7.0+cu110 (torchvision 0.8.0, torchaudio 0.7.0)
```commandline
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/cu110/torch_stable.html
```
Then install dependencies: (if using venv or conda, omit `--user`)
```
pip install --user --upgrade pip
pip install --user torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --user torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --user torch-geometric==1.7.2
pip install --user matplotlib==3.4.3
pip install --user ortools==9.0.9048
```

### Docker Setup (preferred)
Clone this repo and within the repo folder run the following command.

Create image `l2s_image`:
```commandline
sudo docker build -t l2s_image .
```

Create container `l2s_container` from `l2s_image`, and activate it:
```commandline
sudo docker run --gpus all --name l2s_container -it l2s_image
```

## Reproducing
To reproduce the result in the paper, first clone the whole repo:
```
git clone https://github.com/zcaicaros/L2S.git
```
Then run:
```
python3 test_learned.py
```
