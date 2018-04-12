#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys
from time import time

cores = 1
ht    = 2
cpu_offset   = 0
architecture = 'AVX2'
base = sys.argv[1]

create = True
if len(sys.argv) > 2:
    create = False

opt_param_list = [('no_add', ',no_add,')]#, ('no_pad', ',no_pad,'), ('no_add', ',no_add,')]#, ('no_act', ',no_act,'), ('only_add', ',no_lin,no_pad,')]

test_name = filter(None, base.split('/'))[-1]

net_path = os.path.join(base, "net.prototxt")
weights_path = os.path.join(base, "weights.h5")
input_path =  os.path.join(base, "in.h5")
reference_path =  os.path.join(base, "out.h5")

in_file  = h5py.File(input_path)
in_a     = in_file["main"][:]
z = pznet.znet()

if create:
    print ("Creating nets...")
    for name_prefix, opt_flags in opt_param_list:
        znet_path = "/opt/znets/{}_{}cores_opt_{}".format(test_name, cores, name_prefix)
        z.create_net(net_path, weights_path, znet_path, architecture, cores, ht, cpu_offset, opt_flags=opt_flags)
print ("Running nets...")
#system.exit(1)
for name_prefix, opt_flags in opt_param_list:
    znet_path = "/opt/znets/{}_{}cores_opt_{}".format(test_name, cores, name_prefix)
    lib_path  = os.path.join(znet_path, "lib")
    z.load_net(znet_path, lib_path)

    N = 10
    s = time()
    for i in range(N):
        z.forward(in_a)
    e = time()
    print ("{}: {}sec".format(name_prefix, (e - s) / N))



