#!/usr/bin/python
import caffe
import h5py
import numpy as np
import sys
import os

caffe.set_device(1)
caffe.set_mode_gpu()

test_folder = sys.argv[1]
model_path   = os.path.join(test_folder, "net.prototxt")
weights_path = os.path.join(test_folder, "weights.h5")
in_path      = os.path.join(test_folder, "in.h5")
out_path     = os.path.join(test_folder, "out.h5")
print model_path
net = caffe.Net(model_path, 1, weights=weights_path)

in_data = h5py.File(in_path)["main"][:]
net.blobs["input"].data[:] = in_data
net.forward()
out_data = net.blobs["output"].data[...]

out_file = h5py.File(out_path)
out_file.create_dataset('/main', data=out_data)
out_file.close()
