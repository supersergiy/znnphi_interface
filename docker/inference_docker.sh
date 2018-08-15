#!/bin/bash
docker pull seunglab/kaffe:pznet
docker run -v ~/inference_data:/import -v ~/results:/results -v ~/znets:/opt/znets -it seunglab/kaffe:pznet /bin/bash
