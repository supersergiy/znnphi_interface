#!/usr/bin/python
import caffe
import h5py
import numpy as np
import sys
import os
import glob

caffe.set_device(1)
caffe.set_mode_gpu()

def init_weights(t, shape, weights_index, mode):
  if t == 'conv':
    if weights_indes == 0: #kernel
      if mode == 'identity':
        return np.ones(shape) / shape[1]
      elif mode == 'random': #xavier
        return np.random.normal(size=shape, scale=np.sqrt((shape[0] + shape[1]) / 2))
    elif weights_index == 1: #bias
      return np.zeros(shape)
  elif t == 'bn':
    if weights_indes == 0: #scale
      return np.ones(shape) 
    elif weights_index == 1: #mean
      return np.zeros(shape) 
    elif weights_indes == 2: #var
      return np.ones(shape) 
  elif t == 'scale':
    if weights_indes == 0: #mult
      return np.ones(shape) 
    elif weights_index == 1: #bias
      return np.zeros(shape) 

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
      in_data = np.random.random_sample(net.blobs['input'].data.shape)
      #in_data = np.zeros(net.blobs['input'].data.shape)
      in_file.create_dataset('/main', data=in_data)
      print "Dataset '/main' created!"
      in_file.close()

   if not os.path.isfile(weights_path):
      print "Generating weights file..."
      weights_file = h5py.File(weights_path, 'w')
      for p in net.params.keys():
          for i in range(len(net.params[p])):
            if "bn" in p.lower() or "batc" in p.lower():
              layer_type = 'bn'
            elif "sc" in p.lower():
              layer_type = 'scale'
            elif "con" in p.lower():
              layer_type = 'conv'
            else:
              raise Exception("Don't know layer type: {}".format(p))
    
            weights_data = init_weights(layer_type, net.params[p][i].data.shape, i, init_mode) 
            weights_file.create_dataset('/data/{}/{}'.format(p, i), data=weights_data)

      weights_file.close()
