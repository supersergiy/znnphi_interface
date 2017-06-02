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

MODE = 'r'
count = 0
for in_d in inputs:
    for k_d in kernels:
        for ofm in ofms:
            count += 1
            if count > 1000:
                break

            layers  = ["conv"]
            layer_str = "_".join(map(str, layers))
            in_str    = "_".join(map(str, in_d))
            ker_str   = "_".join(map(str, k_d))

            ifm = in_d[1]
            weights_path = "./data/weights/weight_{}_{}_{}_{}.h5".format(ofm, ifm, ker_str, MODE)
            proto_path   = "./nets/{}_{}_x{}_o{}.prototxt".format(layer_str, in_str, ker_str, ofm)
            input_path   = "./data/inputs/input_{}_{}.h5".format(in_str, MODE)
            output_path  = "./data/reference/reference_{}_{}_x{}_o{}_{}.h5".format(layer_str,
                                                                                 in_str,
                                                                                 ker_str,
                                                                                 ofm,
                                                                                 MODE)

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

            net.blobs['input'].data[...] = in_file['input']
            net.forward()

            last_layer_out = layers[-1] + "_o"
            out_file.create_dataset('data', data=net.blobs[last_layer_out].data)

print count
