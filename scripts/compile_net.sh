#!/bin/bash

python scripts/compile_net.py --net ${HOME}/workspace/s1_analysis/13_inference_pznet/model.prototxt --weights ${HOME}/workspace/s1_analysis/13_inference_pznet/train_iter_790000.caffemodel.h5 --cores 8 --ht=2 --output-znet-path /tmp/s1net/core8
