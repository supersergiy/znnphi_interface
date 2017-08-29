#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys

base =     '/home/ubuntu/znnphi_interface/code/znet/reference/unet/inputs'
input_file = 'input.h5'
input_path     = os.path.join(base, input_file)

in_file  = h5py.File(input_path)
in_a     = in_file["main"][0:18, 0:192, 0:192]

znet_path = "/home/ubuntu/tmp/nettynet/"
z = pznet.znet()
z.load_net(znet_path)

for i in range(1):
    out_a    = z.forward(in_a)
