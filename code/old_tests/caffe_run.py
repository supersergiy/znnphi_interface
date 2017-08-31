#!/usr/bin/python
import sys
import h5py
import numpy as np
import caffe
caffe.set_mode_cpu()

proto_path   = sys.argv[1]
input_path   = sys.argv[2]
output_path  = sys.argv[3]
weights_path = sys.argv[4]

net      = caffe.Net(proto_path, caffe.TEST)
in_file  = h5py.File(input_path,   'r')
weights  = h5py.File(weights_path, 'r')
out_file = h5py.File(output_path,  'w')

for layer_p in weights['data']:
	for dataset in weights['data'][layer_p]:
		net.params[layer_p][int(dataset)].data[...] = weights['data'][layer_p][dataset]
net.blobs['input'].data[...] = in_file['input']
net.forward()
out_file.create_dataset('data', data=net.blobs['conv_o'].data)
