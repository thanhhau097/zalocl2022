FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update && \
    apt -y upgrade && \
    apt install -y ffmpeg libsm6 libxext6 enchant myspell-es python3-pip build-essential && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 

RUN apt install -y cmake
RUN pip install -U pip 

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apt install -y git
COPY hoanganh hoanganh
COPY hoanganh-mergetokens hoanganh-mergetokens

