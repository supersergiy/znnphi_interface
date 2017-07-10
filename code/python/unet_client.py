#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys


base =     '/home/ubuntu/znnphi_interface/code/znet/reference/'

net_file = 'unet/unet.prototxt'

weights_file = 'unet/unet.h5'

input_file = 'data/inputs/input.h5'

reference_file = 'data/reference/reference.h5'

net_path       = os.path.join(base, net_file)
weights_path   = os.path.join(base, weights_file)
input_path     = os.path.join(base, input_file)
reference_path = os.path.join(base, reference_file)

in_file  = h5py.File(input_path)
in_a     = in_file["input"][:]

z = pznet.znet(net_path, weights_path)

for i in range(2):
    out_a    = z.forward(in_a)

    reference_file = h5py.File(reference_path)
    reference_a = reference_file["data"][:]
    np.set_printoptions(precision=2)
    diff_a = reference_a - out_a
    error = ssq = np.sum(diff_a**2)
    ref_a = reference_a

    fd = diff_a.flatten()
    fo = out_a.flatten()
    fr = reference_a.flatten()
    boo = np.argmax(fd)

    error = np.sum(diff_a**2)

    if np.isnan(error) or error > 0.010:
        print "Not congrats! Error == {}".format(error)
    else:
        print "Congrats! All pass. Error == {}".format(error)
#out_file = h5py.File(output_path)
#out_file.create_dataset("data", data=out_a)

