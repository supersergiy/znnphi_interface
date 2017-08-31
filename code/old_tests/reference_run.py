#!/usr/bin/python
import sys
import os
import h5py
import numpy as np
import caffe
import json
caffe.set_mode_cpu()

PARAM_PATH = "params.json"
with open(PARAM_PATH, 'rb') as f:
   params = json.load(f)

kernels = params["kernels"]
ofms    = params["ofms"]
inputs  = params["inputs"]
layers  = params["layers"]

MODE = 'r'
count = 0

weights_path = "./data/weights/weights_{}.h5".format(MODE)
proto_path   = "./nets/net.prototxt"
input_path   = "./data/inputs/input.h5"
output_path  = "./data/reference/reference.h5"

print input_path
print output_path

for in_d in inputs:
    for k_d in kernels:
        for ofm in ofms:
            count += 1
            if count > 1000:
                break

            layer_str = "_".join(map(str, layers))
            in_str    = "_".join(map(str, in_d))
            ker_str   = "_".join(map(str, k_d))

            ifm = in_d[1]

            if not os.path.exists(proto_path):
                print "Skipping {}...".format(proto_path)
                continue
            in_file  = h5py.File(input_path,   'r')
            weights  = h5py.File(weights_path, 'r')
            out_file = h5py.File(output_path,  'w')
            net      = caffe.Net(proto_path, caffe.TEST)

            for layer_p in weights['data']:
                for dataset in weights['data'][layer_p]:
                    net.params[layer_p][int(dataset)].data[...] = weights['data'][layer_p][dataset]
	    print in_file['input'][:]
            net.blobs['input'].data[...] = in_file['input'][:]
            net.forward()
            last_layer_out = "output"
	    print net.blobs[last_layer_out].data
            out_file.create_dataset('data', data=net.blobs[last_layer_out].data)

print count
