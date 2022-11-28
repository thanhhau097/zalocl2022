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
COPY hoanganh /code/hoanganh
COPY hoanganh-mergetokens /code/hoanganh-mergetokens
COPY predict.py /code/predict.py
COPY predict.sh /code/predict.sh
COPY predict_notebook.ipynb /code/predict_notebook.ipynb
COPY start_jupyter.sh /code/start_jupyter.sh

RUN pip install transformers datasets scikit-learn
RUN apt install -y zip
RUN pip install jupyterlab

RUN chmod +x /code/predict.sh
RUN chmod +x /code/start_jupyter.sh
WORKDIR /code