#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER      = '/Users/sergiy/seung/caffe_znnphi/'
WEIGHT_FILE_NAME = 'train_iter_300000.caffemodel.h5'

base_filename = DATA_FOLDER + WEIGHT_FILE_NAME
base_file = h5py.File(base_filename, 'r')
base_np = base_file['data']['BatchNorm1']['0']
print base_np.shape

import caffe_pb2
from google.protobuf.text_format import Merge
net = caffe_pb2.NetParameter()
Merge((open("deploy.prototxt",'r').read()), net)
print (net.layer[10].blobs)
for l in net.layer:
	print l
	break
'''
for i in range(0, len(STRIDE_LIST)):
        s = STRIDE_LIST[i]
        s_p= [float(s)/ 100.0]*3
        s_filename = DATA_FOLDER + H5_FILE_TEMPLATE.format(*([s]*3))
        s_file = h5py.File(s_filename, 'r')
        s_np = s_file['main'][()]'''
