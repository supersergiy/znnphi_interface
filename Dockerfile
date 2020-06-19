FROM ubuntu:20.04
LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng.wu@gmail.com"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV HOME "/root"
# prevent interaction with tzdata
ENV DEBIAN_FRONTEND noninteractive
ENV ZNNPHI_PATH "/root/workspace/pznet"
ENV INTEL_LIB_PATH "/opt/intel/lib/intel64"
ENV INTEL_LIB_URL "https://github.com/seung-lab/pznet/releases/download/v0.2.0"
ENV PYTHONPATH "${ZNNPHI_PATH}/python"
ENV LD_LIBRARY_PATH "${INTEL_LIB_PATH}"


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
        tar \
        wget && \
    echo "set up mkl libraries..." && \
    mkdir -p /opt/intel/lib/intel64 && \
    wget --quiet "${INTEL_LIB_URL}/libimf.so" -O "${INTEL_LIB_PATH}/libimf.so" && \
    wget --quiet "${INTEL_LIB_URL}/libintlc.so" -O "${INTEL_LIB_PATH}/libintlc.so" && \
    wget --quiet "${INTEL_LIB_URL}/libintlc.so.5" -O "${INTEL_LIB_PATH}/libintlc.so.5" && \
    wget --quiet "${INTEL_LIB_URL}/libirng.so" -O "${INTEL_LIB_PATH}/libirng.so" && \
    wget --quiet "${INTEL_LIB_URL}/libsvml.so" -O "${INTEL_LIB_PATH}/libsvml.so" && \
    wget --quiet "${INTEL_LIB_URL}/libiomp5.so" -O "${INTEL_LIB_PATH}/libiomp5.so" && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    pip install -U pip && \
    pip install -r requirements.txt && \
    # clean up the apt installation
    apt-get clean && \
    apt-get autoremove --purge -y && \
    rm -rf /var/lib/apt/lists/* 

# echo "activate the conda environment..." && \
#RUN python -c "from pznet.pznet import PZNet" && \
#    python scripts/compile_net.py --help 


WORKDIR "${ZNNPHI_PATH}/scripts"

