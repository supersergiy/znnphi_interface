#!/usr/bin/python
import sys
import h5py
import numpy as np
import caffe
caffe.set_mode_cpu()

proto_path  = sys.argv[1]
input_path  = sys.argv[2]
output_path = sys.argv[3]

net      = caffe.Net(proto_path, caffe.TEST)
in_file  = h5py.File(input_path, 'r')
out_file = h5py.File(output_path, 'w')


net.blobs['data'] = in_file
net.forward()
#print net.blobs['result'].data
out_file.create_dataset('data', data=net.blobs['result'].data)
