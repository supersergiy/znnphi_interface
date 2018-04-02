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
  if t == 'deconv': 
    if weights_index == 0: #kernel
      
      if shape[3] == 4: #this means it's a bilinear
        fan_in  = 1  * (shape[2] * shape[3] * shape[4])
        fan_out = 1 * (shape[2] * shape[3] * shape[4])
      else:
        fan_in  = shape[0]  * (shape[2] * shape[3] * shape[4])
        fan_out = shape[1] * (shape[2] * shape[3] * shape[4])
      std_dev = np.sqrt(2.0 / (fan_in + fan_out))
      results = np.random.normal(size=shape, scale=std_dev) #std_var/ (shape[2] * shape[3] * shape[4])
      return results
    elif weights_index == 1: #bias
      return np.zeros(shape)
  elif t == 'conv': 
    if weights_index == 0: #kernel
      if mode == 'identity':
        result = np.ones(shape) / (shape[0] * shape[2] * shape[3] * shape[4])
      elif mode == 'random': #xavier
        #std_var = np.sqrt((shape[0] + shape[1]) / 2)
        fan_in = shape[0]  * (shape[2] * shape[3] * shape[4])
        fan_out = shape[1] * (shape[2] * shape[3] * shape[4])
	std_dev = np.sqrt(2.0 / (fan_in + fan_out))
        result = np.random.normal(size=shape, scale=std_dev) 
      return result
    elif weights_index == 1: #bias
      return np.zeros(shape)
  elif t == 'bn':
    if weights_index == 0: #variance
      return np.zeros(shape) 
    elif weights_index == 1: #mean
      return np.ones(shape) 
    elif weights_index == 2: #scale
      return np.ones(shape) 
  elif t == 'scale':
    if weights_index == 0: #mult
      return np.ones(shape) 
    elif weights_index == 1: #bias
      return np.zeros(shape) 

file_spec = sys.argv[1]
test_list = glob.glob(file_spec)
init_mode = 'random'

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
      #in_data = np.ones(net.blobs['input'].data.shape)
      in_file.create_dataset('/main', data=in_data)
      print "Dataset '/main' created!"
      in_file.close()

   if not os.path.isfile(weights_path):
      print "Generating weights file..."
      weights_file = h5py.File(weights_path, 'w')
      for p in net.params.keys():
          for i in range(len(net.params[p])):
	    if "deconv" in p.lower():
              layer_type = 'deconv'
            elif "bn" in p.lower() or "batc" in p.lower():
              layer_type = 'bn'
            elif "sca" in p.lower():
              layer_type = 'scale'
            elif "con" in p.lower() or "aff" in p.lower():
              layer_type = 'conv'
            else:
              raise Exception("Don't know layer type: {}".format(p))
    
            weights_data = init_weights(layer_type, net.params[p][i].data.shape, i, init_mode) 
            weights_file.create_dataset('/data/{}/{}'.format(p, i), data=weights_data)

      weights_file.close()
