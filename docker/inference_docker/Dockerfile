# Use an official Python runtime as a parent image
FROM ubuntu 

# Set the working directory to /app
WORKDIR /opt

# Copy the current directory contents into the container at /app
ADD . /opt
RUN mv intel.tar.gz / && cd / && tar -xzf intel.tar.gz

RUN apt-get update && \
      apt-get -y install sudo

RUN sudo apt-get install -y python
RUN sudo apt-get install -y python-pip
RUN sudo apt-get install -y git
RUN sudo apt-get install -y vim
RUN sudo apt-get install -y python-tk

RUN pip install -r requirements.txt

#get znet and kaffe repos
RUN git clone https://github.com/supersergiy/znnphi_interface.git
RUN git clone https://github.com/torms3/kaffe.git
RUN mkdir /opt/.tmp

RUN tar -xzf lib.tar.gz
RUN tar -xzf znets.tar.gz

RUN cd kaffe && git checkout cpu_inference
RUN cd znnphi_interface 

ENV LD_LIBRARY_PATH /opt/intel/lib/intel64
ENV ZNNPHI_PATH /opt/znnphi_interface 
ENV PYTHONPATH /opt/znnphi_interface/code/znet/src/python:/opt/kaffe 

