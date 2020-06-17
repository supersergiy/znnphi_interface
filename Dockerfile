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
        wget && \
    echo "setting up minicoda..." && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    echo "downloaded miniconda, start installing..." && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    #/opt/conda/bin/conda init bash && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo "set up mkl libraries..." && \
    mkdir -p /opt/intel/lib/intel64 && \ 
    # clean up the apt installation
    apt-get clean && \
    apt-get autoremove --purge -y && \
    rm -rf /var/lib/apt/lists/* && \
    echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
    echo "export PYTHONPATH=${ZNNPHI_PATH}/python:$PYTHONPATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    source ~/.bashrc && \
    

# echo "activate the conda environment..." && \
RUN conda env create -f environment.yml && \
    conda activate pznet && \
    python -c "from pznet.pznet import PZNet" && \
    python scripts/compile_net.py --help 


WORKDIR "${ZNNPHI_PATH}/scripts"

