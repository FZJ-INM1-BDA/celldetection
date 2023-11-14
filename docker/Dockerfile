# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

RUN pip install --upgrade pip
RUN pip3 install safetensors==0.2.8

RUN apt-get update && apt-get install libgl1 -y

COPY ./celldetection ./celldetection
COPY ./setup.py ./
COPY ./requirements.txt ./
COPY ./MANIFEST.in ./
COPY ./README.md ./
COPY ./LICENSE ./
COPY ./scripts/*.py ./

RUN pip3 install .
RUN pip3 install psutil
RUN pip3 install imagecodecs

RUN mkdir /inputs
RUN mkdir /outputs
RUN mkdir /models
