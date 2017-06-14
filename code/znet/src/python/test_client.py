#!/usr/bin/python
import pznet
from operator import mul
import os
import numpy as np
import h5py
import sys


input_dim    = [1, 1, 1, 10, 10]
ofm          = 1
kernel_dim   = [1, 1, 1]
layers       = ["conv", "elu"]

layers_prefix = '_'.join(layers)
base =     '/home/ubuntu/znnphi_interface/code/znet/reference/'
net_file = 'nets/test.prototxt'

weights_file = 'data/weights/weights_{}_{}_{}_{}_{}_r.h5'.format(
                                                                ofm,
                                                                input_dim[1], #ifm
                                                                kernel_dim[0],
                                                                kernel_dim[1],
                                                                kernel_dim[2])

net_path       = os.path.join(base, net_file)
weights_path   = os.path.join(base, weights_file)

z = pznet.znet(net_path, weights_path)
n = reduce(mul, input_dim)

in_a     = np.zeros(n, dtype=np.float)
out_a    = z.forward(in_a)
print out_a
import pdb; pdb.set_trace()

