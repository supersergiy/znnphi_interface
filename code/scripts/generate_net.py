#!/usr/bin/python3
import pznet
import os
import numpy as np
import h5py
import sys

cores = 2

base = sys.argv[1]
test_name = base.split('/')[-1]

net_path = os.path.join(base, "net.prototxt")
weights_path = os.path.join(base, "weights.h5")
input_path =  os.path.join(base, "in.h5")
reference_path =  os.path.join(base, "out.h5")

in_file  = h5py.File(input_path)
in_a     = in_file["main"][:]

znet_path = "/opt/znets/{}_{}cores".format(test_name, cores)
z = pznet.znet()
z.create_net(net_path, weights_path, znet_path, cores)
