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
	net = caffe.Net(model_path, 1)
	import pdb; pdb.set_trace()

	out_file = h5py.File(out_path)
	out_file.create_dataset('/main', data=out_data)
	out_file.close()
