FROM seunglab/icpc:2017v3 

WORKDIR /opt
RUN apt-get update 
RUN apt-get install -y python python-pip git vim python-tk \
		       screen wget vim 

#install Docker
RUN wget -qO- https://get.docker.com/ | sh && groupadd docker; usermod -aG docker root && newgrp docker 

#get znet and kaffe repos
RUN git clone https://github.com/supersergiy/znnphi_interface.git
RUN git clone https://github.com/torms3/kaffe.git
RUN mkdir /opt/.tmp

RUN cd kaffe && git checkout cpu_inference

ENV LD_LIBRARY_PATH /opt/intel/lib/intel64
ENV ZNNPHI_PATH /opt/znnphi_interface 
ENV PYTHONPATH /opt/znnphi_interface/code/znet/src/python:/opt/kaffe 

ADD ./pz_files/ /opt
RUN mv intel.tar.gz / && cd / && tar -xzf intel.tar.gz

RUN pip install -r requirements.txt
RUN tar -xzf lib.tar.gz
RUN tar -xzf znets.tar.gz
RUN mv /opt/.screenrc /root/
RUN cd /opt/znnphi_interface \
    && git config --global user.email "popovych@princeton.edu" \
    && git config credential.helper store

