#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys
from time import time

cores = 2
ht    = 1
cpu_offset   = 0
architecture = 'AVX512'
base = sys.argv[1]

create = True
if len(sys.argv) > 2:
    create = False

test_name = filter(None, base.split('/'))[-1]

net_path = os.path.join(base, "net.prototxt")
weights_path = os.path.join(base, "weights.h5")
input_path =  os.path.join(base, "in.h5")
reference_path =  os.path.join(base, "out.h5")

in_file  = h5py.File(input_path)
in_a     = np.random.rand(22,224,224)
znet_path = "/opt/znets/{}_{}cores".format(test_name, cores)
lib_path  = os.path.join(znet_path, "lib")
z = pznet.znet()
if create:
    print "Creating net..."
    z.create_net(net_path, weights_path, znet_path, architecture, cores, ht, cpu_offset)
#sys.exit(1)
print "Running net..."
z.load_net(znet_path, lib_path)

N = 20
s = time()
for i in range(N):
    z.forward(in_a)
e = time()
print (e - s) / N

#out_file = h5py.File(output_path)
#out_file.create_dataset("data", data=out_a)

