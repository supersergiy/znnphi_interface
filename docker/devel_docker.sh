#!/bin/bash
docker pull seunglab/pznet:devel
docker run -v ~/inference_data:/import -v ~/znets:/opt/znets -v /opt/intel/licenses:/opt/intel/licenses -v ~/results:/results -v /home/ubuntu/znnphi_interface/code/test/tests:/tests -it seunglab/pznet:devel /bin/bash
