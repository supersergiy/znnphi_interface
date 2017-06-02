#!/usr/bin/python
import h5py
import numpy as np
import sys
import json

PARAM_PATH = "params.json"
FIRST_SPACIAL_D = 2

with open(PARAM_PATH, 'rb') as f:
	params = json.load(f)

kernels = params["kernels"] 
ofms    = params["ofms"]
inputs  = params["inputs"]

count = 0
for in_d in inputs:
	for k_d in kernels:
		for ofm in ofms:
			count += 1
			bad_param = False
			for i in range(len(k_d)):
				if (in_d[FIRST_SPACIAL_D + i] < k_d[i]):
				 	bad_param = True	
		        if bad_param:
				continue		
			layers  = ["conv"]
			layer_str = "_".join(map(str, layers))
			in_str    = "_".join(map(str, in_d))
			ker_str    = "_".join(map(str, k_d))

			out_path = "./nets/{}_{}_x{}_o{}.prototxt".format(layer_str, in_str, ker_str, ofm)
	
			with open(out_path, 'w') as out_f:
				#base
				base = open("./templates/base.prototxt").read()
				for i in range(len(in_d)):
					base = base.replace("[IN{}]".format(i + 1), str(in_d[i]))	
				out_f.write(base)

				#conv
				conv = open("./templates/{}.prototxt".format("conv")).read()
				for i in range(len(k_d)):	
					conv = conv.replace("[K{}]".format(i + 1), str(k_d[i]))
				conv = conv.replace("[OFM]", str(ofm))
				out_f.write(conv)
