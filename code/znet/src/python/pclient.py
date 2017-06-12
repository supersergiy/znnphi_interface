#!/usr/bin/python
import pznet
import os
import numpy as np


base =     '/home/ubuntu/znnphi_interface/code/znet/reference/'
net_file = 'nets/conv_1_8_18_192_192_x3_1_1_o8.prototxt'
weights_file = 'data/weights/weight_8_8_3_1_1_r.h5'

net_path     = os.path.join(base, net_file)
weights_path = os.path.join(base, weights_file)

z = pznet.znet(net_path, weights_path)
dim = [1,8, 18, 192,192]
in_a = np.ones(dim, dtype=np.float)
out_a = z.forward(in_a)
print out_a
