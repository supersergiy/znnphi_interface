#!/usr/bin/python
import caffe
import h5py
import numpy as np

caffe.set_device(1)
caffe.set_mode_gpu()

model_path   = "/sergiy_shared/overlap/deploy.prototxt"
weights_path = "/sergiy_shared/overlap/train_iter_1000000.caffemodel.h5"
in_path      = "/sergiy_shared/znnphi_interface/code/znet/reference/patch1.h5"
out_path     = "/sergiy_shared/znnphi_interface/code/znet/reference/unet_out.h5"

in_data = h5py.File(in_path)["main"][...]

net = caffe.Net(model_path, weights_path, caffe.TEST)
net.blobs["input"].data[...] = in_data
net.forward()
out_data = net.blobs["affinity"].data[...]

out_file = h5py.File(out_path)
out_file.create_dataset('/main', data=out_data)
out_file.close()
