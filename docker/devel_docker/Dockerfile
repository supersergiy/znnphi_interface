FROM seunglab/icpc:2017v3 

WORKDIR /opt
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python-software-properties -y && apt-get install apt-file -y \ 
    && apt-file update && apt-get install software-properties-common -y \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y &&  apt-get update \
    && apt-get install gcc-snapshot -y && apt-get update && pt-get install gcc-7 g++-7 -y
RUN apt-get install -y python python-pip git python-tk \
        		       screen wget vim 


#get znet and kaffe repos
RUN ls
RUN git clone https://github.com/supersergiy/znnphi_interface.git
RUN git clone https://github.com/torms3/kaffe.git
RUN git clone https://bitbucket.org/poozh/ond-winograd.git 
RUN mkdir /opt/.tmp

RUN cd kaffe && git checkout cpu_inference
#Using this instead of ENV so that can work within screen
RUN echo "export LD_LIBRARY_PATH=/opt/intel/lib/intel64" > /root/.bashrc
RUN echo "export ZNNPHI_PATH=/opt/znnphi_interface" >> /root/.bashrc 
RUN echo "export PYTHONPATH=/opt/znnphi_interface/code/znet/src/python:/opt/kaffe" >> /root/.bashrc
RUN . /root/.bashrc

ADD ./pz_files/ /opt
RUN mv intel.tar.gz / && cd / && tar -xzf intel.tar.gz

RUN pip install -r requirements.txt
RUN tar -xzf lib.tar.gz
RUN tar -xzf znets.tar.gz
RUN mv /opt/.screenrc /root/
RUN cd /opt/znnphi_interface \
    && git config --global user.email "popovych@princeton.edu" \
    && git config credential.helper store

RUN git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime \
    && sh ~/.vim_runtime/install_awesome_vimrc.sh
