FROM seunglab/icpc:2016v4 


WORKDIR /opt
RUN apt-get update 
RUN apt-get install -y python python-pip git vim python-tk \
		       screen wget vim cpio

#get znet and kaffe repos
RUN pwd 
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
RUN git config --global user.email "popovych@princeton.edu"
RUN wget -qO- https://get.docker.com/ | sh && sudo groupadd docker; sudo usermod -aG docker root && newgrp docker 
