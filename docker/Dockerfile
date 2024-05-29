# syntax=docker/dockerfile:1

FROM python:3.10-buster

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install safetensors==0.2.8

RUN apt-get update && apt-get install libgl1 mpich -y
RUN pip install mpi4py

COPY ./celldetection ./celldetection
COPY ./setup.py ./
COPY ./requirements.txt ./
COPY ./MANIFEST.in ./
COPY ./README.md ./
COPY ./LICENSE ./
COPY ./celldetection_scripts ./celldetection_scripts

RUN pip install imagecodecs; exit 0  # optional due to unresolved issues on ARM
RUN pip install .
RUN pip install psutil

RUN mkdir /inputs
RUN mkdir /outputs
RUN mkdir /models
