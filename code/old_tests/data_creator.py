#!/usr/bin/python
import h5py
import numpy as np
import sys
import json

FILLER = 'r'

with open("params.json", 'rb') as f:
   params = json.load(f)

kernels = params["kernels"]
ofms    = params["ofms"]
inputs  = params["inputs"]
layers  = params["layers"]

in_d = inputs[0]
k    = kernels[0]
ofm  = ofms[0]

def generate(dim):
  if FILLER == 'r':
     return np.random.uniform(low=-2.0, high=2.0, size=dim)
  else:
     return np.array(range(np.prod(dim))).reshape(dim)

def generate_layer_weights(lname, ltype, f, in_d, ofm):
    prefix = "data/{}/".format(lname)

    ifm = in_d[1]
    if ltype == "conv":

        dim0 = (ofm, ifm, k[0], k[1], k[2])
        dim1 = (ofm,)

        name0 = "0"
        name1 = "1"

        dset0 = f.create_dataset(prefix + name0, dim0, dtype='f')
        dset1 = f.create_dataset(prefix + name1, dim1, dtype='f')

        dset0[...] = generate(dim0)
        dset1[...] = generate(dim1)

    if ltype == "scale":
        dim0 = (ofm,)
        dim1 = (ofm,)

        name0 = "0"
        name1 = "1"

        dset0 = f.create_dataset(prefix + name0, dim0, dtype='f')
        dset1 = f.create_dataset(prefix + name1, dim1, dtype='f')

        dset0[...] = generate(dim0)
        dset1[...] = generate(dim1)

    if ltype == "bnorm":
        dim0 = (ifm,)
        dim1 = (ifm,)
        dim2 = (1,)

        name0 = "0"
        name1 = "1"
        name2 = "2"

        dset0 = f.create_dataset(prefix + name0, dim0, dtype='f')
        dset1 = f.create_dataset(prefix + name1, dim1, dtype='f')
        dset2 = f.create_dataset(prefix + name2, dim2, dtype='f')

        dset0[...] = generate(dim0)
        dset1[...] = generate(dim1) + 2.0
        dset2[...] = np.ones(dim2)#generate(dim2) 


def generate_all_weights():
      out_file_path = "data/weights/weights.h5".format(FILLER)
      f = h5py.File(out_file_path, "w")

      for i in range(len(layers)):
          generate_layer_weights("{}_{}".format(layers[i], i), layers[i], f, in_d, ofm)


def generate_inputs():
       dim0 = in_d

       suffix = "_".join(map(str, dim0))
       prefix = ""

       out_file_path = "data/inputs/input.h5".format(suffix, FILLER)
       name0 = "input"

       f = h5py.File(out_file_path, "w")
       dset0 = f.create_dataset(prefix + name0, dim0, dtype='f')

       dset0[...] = generate(dim0)

generate_all_weights()
generate_inputs()
