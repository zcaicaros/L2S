FROM python:3.8.10

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu1804-11-4-local_11.4.2-470.57.02-1_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu1804-11-4-local_11.4.2-470.57.02-1_amd64.deb
RUN sudo apt-key add /var/cuda-repo-ubuntu1804-11-4-local/7fa2af80.pub
RUN sudo apt-get update
RUN sudo apt-get -y install cuda