FROM ubuntu 

ADD . /opt
WORKDIR /opt

RUN apt-get update && \
      apt-get -y install sudo

RUN sudo apt-get install \
       --no-install-recommends -y python python-pip git vim python-tk \
	                          libprotobuf-dev libleveldb-dev libsnappy-dev \
                                  libopencv-dev libhdf5-serial-dev protobuf-compiler \
 			          libgflags-dev libgoogle-glog-dev liblmdb-dev \ 
			          libatlas-base-dev libboost-all-dev cmake
RUN sudo apt-get install gcc
RUN pip install numpy 

#install pycaffe
#RUN git clone https://github.com/torms3/caffe.git

RUN git clone https://github.com/BVLC/caffe.git
RUN sed -i '33d' caffe/CMakeLists.txt 
RUN mkdir caffe/build && cd caffe/build  
RUN cd caffe/build && cmake ..; return 0
RUN cd caffe/build && make all && make install 

RUN python -c 'import caffe'
#ENV PYTHONPATH /opt/znnphi_interface/code/znet/src/python:/opt/kaffe:/opt/caffe/python/caffe
#RUN python -c'import caffe'
#get znet and kaffe repos
#RUN git clone https://github.com/supersergiy/znnphi_interface.git
#RUN git clone https://github.com/torms3/kaffe.git
#RUN mkdir /opt/.tmp

#RUN cd kaffe && git checkout cpu_inference

#ENV LD_LIBRARY_PATH /opt/intel/lib/intel64
#ENV ZNNPHI_PATH /opt/znnphi_interface 
#ENV PYTHONPATH /opt/znnphi_interface/code/znet/src/python:/opt/kaffe:/opt/caffe/python

#RUN mv intel.tar.gz / && cd / && tar -xzf intel.tar.gz
#RUN tar -xzf lib.tar.gz
#RUN tar -xzf znets.tar.gz
#RUN pip install -r requirements.txt
