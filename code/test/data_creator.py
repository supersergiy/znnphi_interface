#!/usr/bin/python
import h5py
import numpy as np
import sys
import json

FILLER = sys.argv[1] 
MODE = sys.argv[2] 

with open("params.json", 'rb') as f:
	params = json.load(f)

kernels = params["kernels"] 
ofms    = params["ofms"]
inputs  = params["inputs"] 

if MODE == "w":
	for in_d in inputs:
		for k in kernels:
			for ofm in ofms:
				ifm = in_d[1]

				dim0 = (ofm, ifm, k[0], k[1], k[2])
				dim1 = (ofm,)	

				suffix = "_".join(map(str, dim0))
				prefix = "data/conv/"

				out_file_path = "data/weights/weight_{}_{}.h5".format(suffix, FILLER)
				name0 = "0"
				name1 = "1"

				f = h5py.File(out_file_path, "w")
				dset0 = f.create_dataset(prefix + name0, dim0, dtype='f')
				dset1 = f.create_dataset(prefix + name1, dim1, dtype='f')

				if FILLER == 'r':
					dset0[...] = np.random.rand(*dim0)
					dset1[...] = np.random.rand(*dim1)
				else:
					dset0[...] = np.array(range(np.prod(dim0))).reshape(dim0)
					dset1[...] = np.array(range(np.prod(dim1))).reshape(dim1)
if MODE == "i":
	for in_d in inputs:
			dim0 = in_d 

			suffix = "_".join(map(str, dim0))
			prefix = ""

			out_file_path = "data/inputs/input_{}_{}.h5".format(suffix, FILLER)
			name0 = "input"

			f = h5py.File(out_file_path, "w")
			dset0 = f.create_dataset(prefix + name0, dim0, dtype='f')

			if FILLER == 'r':
				dset0[...] = np.random.rand(*dim0)
			else:
				dset0[...] = np.array(range(np.prod(dim0))).reshape(dim0)
			#dset1[...] = np.random.rand(*dim1)
