#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys


base =     '/home/ubuntu/znnphi_interface/code/znet/reference/'

net_file = 'unet/unet.prototxt'

weights_file = 'unet/unet.h5'

input_file = 'unet/inputs/input.h5'

reference_file = 'unet/reference/reference.h5'

net_path       = os.path.join(base, net_file)
weights_path   = os.path.join(base, weights_file)
input_path     = os.path.join(base, input_file)
reference_path = os.path.join(base, reference_file)

in_file  = h5py.File(input_path)
in_a     = in_file["input"][:]

#z.load_net("/home/ubuntu/znnphi_interface/code/znet/src/python/pznet/.tmp")

z = pznet.znet()
z.create_net(net_path, weights_path)
for i in range(1):
    out_a    = z.forward(in_a)

    reference_file = h5py.File(reference_path)
    reference_a = reference_file["output"][:]
    np.set_printoptions(precision=2)

    diff_a = reference_a - out_a                                                                                                                           
    rel_d = np.abs(diff_a) / (out_a + 0.0000000001)
    mask1 = diff_a > 1e-5
    mask2 = rel_d > 1e-5
 
    fd = diff_a.flatten()
    fo = out_a.flatten()
    fr = reference_a.flatten()

    errors = rel_d * mask1 * mask2 * reference_a
    error = np.sum((errors*10)**2)

    max_d = np.max(np.abs(diff_a))
    i = np.argmax(np.abs(diff_a.flatten()))
    max_rel_d = max_d / fr[i]
    print "Max rel d: {}".format(max_rel_d)
    print "Max d: {}".format(max_d)
    print "Average d: {}".format(np.average(rel_d))

    if np.isnan(error):
        print "Not congrats! Error == {}".format(error)
        #import pdb; pdb.set_trace()
    elif error > 0.010:
        print "Not congrats! Error == {}".format(error)
        import pdb; pdb.set_trace()
    else:
        print "Congrats! All pass. Error == {}".format(error)
#out_file = h5py.File(output_path)
#out_file.create_dataset("data", data=out_a)

