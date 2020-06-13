#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys


base =     '/home/ubuntu/znnphi_interface/code/znet/reference/'

net_file = 'nets/net.prototxt'

weights_file = 'data/weights/weights_r.h5'

input_file = 'data/inputs/input.h5'

net_path       = os.path.join(base, net_file)
weights_path   = os.path.join(base, weights_file)
input_path     = os.path.join(base, input_file)

z = pznet.znet(net_path, weights_path)

np.set_printoptions(precision=2)
in_file  = h5py.File(input_path)
in_a     = np.random.rand(1, 1, 18, 192, 192)
out_a    = z.forward(in_a)

fo = out_a.flatten()
import pdb; pdb.set_trace()

