#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys

test_name = sys.argv[1]
cores = 2 

base = os.path.join('/home/ubuntu/znnphi_interface/code/test/tests', test_name)
net_path = os.path.join(base, "net.prototxt")
weights_path = os.path.join(base, "weights.h5")

znet_path = "/home/ubuntu/znets/{}_{}cores".format(test_name, cores)
z = pznet.znet()
z.create_net(net_path, weights_path, znet_path, cores)
