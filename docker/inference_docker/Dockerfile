FROM ubuntu:16.04 

WORKDIR /opt

RUN apt-get update 
RUN apt-get install \
       --no-install-recommends -y git vim python-tk \
	                          libprotobuf-dev libleveldb-dev libsnappy-dev \
                                 libopencv-dev libhdf5-serial-dev protobuf-compiler \
 			          libgflags-dev libgoogle-glog-dev liblmdb-dev \ 
			          libatlas-base-dev libboost-all-dev cmake gcc g++ ocl-icd-opencl-dev \
				  opencl-headers clinfo ocl-icd-libopencl1
RUN apt-get install python-pip -y
RUN pip install --upgrade
RUN pip install numpy scikit-image scipy protobuf 
RUN pip install protobuf_to_dict
RUN pip install tifffile h5py
RUN apt-get install h5utils -y

#get pznet and kaffe repos
RUN git clone https://github.com/supersergiy/znnphi_interface.git
RUN git clone https://github.com/torms3/kaffe.git
RUN mkdir /opt/.tmp

RUN cd kaffe && git checkout smooth_integration 

ENV LD_LIBRARY_PATH /opt/intel/lib/intel64
ENV ZNNPHI_PATH /opt/znnphi_interface 
ENV PYTHONPATH $PYTHONPATH:/opt/znnphi_interface/code/znet/src/python:/opt/kaffe

ADD export/ /opt/
RUN mv /opt/intel.tar.gz / && cd / && tar -xzf intel.tar.gz
