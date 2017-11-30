#!/usr/bin/python
import caffe
import h5py
import numpy as np
import sys
import os
import glob

caffe.set_device(2)
caffe.set_mode_gpu()

file_spec = sys.argv[1]
test_list = glob.glob(file_spec)
for test_folder in test_list:
   model_path   = os.path.join(test_folder, "net.prototxt")
   weights_path = os.path.join(test_folder, "weights.h5")
   in_path      = os.path.join(test_folder, "in.h5")
   out_path     = os.path.join(test_folder, "out.h5")
   print model_path
   net = caffe.Net(model_path, caffe.TEST)

   if not os.path.isfile(in_path):
      print "Generating input file..."
      in_file = h5py.File(in_path, 'w')
      in_file.create_dataset('/main', data=np.ones(net.blobs['input'].data.shape))
      print "Dataset '/main' created!"
      in_file.close()

   if not os.path.isfile(weights_path):
      print "Generating weights file..."
      weights_file = h5py.File(weights_path, 'w')
      for p in net.params.keys():
         for i in range(len(net.params[p])):
            #net.params[p][i].data[:] = 1.0
            weights_file.create_dataset('/data/{}/{}'.format(p, i), data=net.params[p][i].data[:])

      weights_file.close()
