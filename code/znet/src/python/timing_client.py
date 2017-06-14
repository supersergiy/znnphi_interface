#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys


net_path       = "/home/ubuntu/znnphi_interface/code/znet/nets/unet.prototxt"
weights_path       = "/home/ubuntu/znnphi_interface/code/znet/nets/unet.h5"

z = pznet.znet(net_path, weights_path)

in_dim   = [1, 1, 18, 192, 192]
in_a     = np.zeros(in_dim, dtype=np.float).flatten()
out_a    = z.forward(in_a)

