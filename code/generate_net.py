#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys


base =    sys.argv[1] 
net_file = 'net.prototxt'
weights_file = 'weights.h5'

net_path       = os.path.join(base, net_file)
weights_path   = os.path.join(base, weights_file)
#net_path = "/home/ubuntu/new_unet/unet.prototxt"
#weights_path = "/home/ubuntu/new_unet/unet.h5"

znet_path = "/home/ubuntu/znets/last_net"
z = pznet.znet()
z.create_net(net_path, weights_path, znet_path)
