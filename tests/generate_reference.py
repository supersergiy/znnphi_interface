#!/usr/bin/python
import caffe
import h5py
import numpy as np
import sys
import os
import glob
from time import time 

caffe.set_device(1)
caffe.set_mode_gpu()

file_spec = sys.argv[1]
test_list = glob.glob(file_spec)

for test_folder in test_list:
   model_path   = os.path.join(test_folder, "net.prototxt")
   weights_path = os.path.join(test_folder, "weights.h5")
   in_path      = os.path.join(test_folder, "in.h5")
   out_path     = os.path.join(test_folder, "out.h5")
   print model_path
   net = caffe.Net(model_path, 1, weights=weights_path)

   in_data = h5py.File(in_path)["main"][:]
   net.blobs["input"].data[:] = in_data
   net.forward()
   net.forward()
   s = time()
   net.forward()
   net.forward()
   net.forward()
   net.forward()
   net.forward()
   e = time()

   print ("{} sec".format((e - s) / 5))
   out_data = net.blobs["output"].data[...]
   print ("mean: {}".format(np.mean(out_data)))
   out_file = h5py.File(out_path)
   out_file.create_dataset('/main', data=out_data)
   out_file.close()
