FROM ubuntu:20.04
LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng.wu@gmail.com"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# prevent interaction with tzdata
ENV DEBIAN_FRONTEND noninteractive
ENV ZNNPHI_PATH "${HOME}/workspace/pznet"
ENV LD_LIBRARY_PATH "/opt/intel/lib/intel64":$LD_LIBRARY_PATH

WORKDIR "${ZNNPHI_PATH}"
COPY . .

RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends \
        apt-utils \
        bzip2 \
        ca-certificates \
        libxext6 \
        libsm6 \
        libxrender1 \
        hdf5-tools \
        python3-dev \
        python3-pip \
        wget && \
    echo "set up mkl libraries..." && \
    mkdir -p /opt/intel/lib/intel64 && \ 
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    pip install -U pip && \
    pip install -r requirements.txt && \
    # clean up the apt installation
    apt-get clean && \
    apt-get autoremove --purge -y && \
    rm -rf /var/lib/apt/lists/* && \
    echo "export PYTHONPATH=${ZNNPHI_PATH}/python:$PYTHONPATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH" >> ~/.bashrc 

# echo "activate the conda environment..." && \
#RUN python -c "from pznet.pznet import PZNet" && \
#    python scripts/compile_net.py --help 


WORKDIR "${ZNNPHI_PATH}/scripts"

