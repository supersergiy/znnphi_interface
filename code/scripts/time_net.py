#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys
from time import time
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-b", "--base", dest="base_path")
parser.add_option("-i", "--input_mode", dest="input_mode", default="read")
parser.add_option("--iter", dest="num_iter", default=20)
parser.add_option("-c", "--cores", dest="conv_cores", default=2)
parser.add_option("-O", dest="optimization", default="full_opt")
parser.add_option("--ht", dest="conv_ht", default=2)
parser.add_option("--act_cores", dest="act_cores", default=-1)
parser.add_option("--act_ht", dest="act_ht", default=-1)
parser.add_option("--lin_cores", dest="lin_cores", default=-1)
parser.add_option("--lin_ht", dest="lin_ht", default=-1)
parser.add_option("--recompile", action="store_true", dest="recompile", default=False)
parser.add_option("--dont_run", action="store_false", dest="run", default=True)
parser.add_option("--dont_run", action="store_false", dest="run", default=True)
parser.add_option("--ignore", action="append", dest="ignore", default=[])

parser.add_option("--arch", dest="architecture", default="AVX2",
        help="The cpu architexture: {AVX2, AVX512}")
(options, args) = parser.parse_args()

cpu_offset   = 0
architecture = options.architecture
base         = options.base_path
recompile    = options.recompile
optimization = options.optimization
input_mode   = options.input_mode
N            = options.num_iter
ignore       = ','.join(options.ignore)

core_options = {}
core_options["conv"] = [options.conv_cores, options.conv_ht]
core_options["act"]  = [options.act_cores, options.act_ht]
core_options["lin"]  = [options.lin_cores, options.lin_ht]

test_name = filter(None, base.split('/'))[-1]

net_path = os.path.join(base, "net.prototxt")
weights_path = os.path.join(base, "weights.h5")
input_path =  os.path.join(base, "in.h5")

z = pznet.znet()
znet_path = "/opt/znets/{}_{}cores_{}".format(test_name, core_options["conv"][0], optimization)
lib_path  = os.path.join(znet_path, "lib")
if recompile:
    print ("Recompiling...")
    z.create_net(net_path, weights_path, znet_path, architecture, core_options, cpu_offset, optimization, ignore)

if options.run:
    print "Loading net..."
    z.load_net(znet_path, lib_path)

    in_shape = z.in_shape()
    if input_mode == 'read':
        in_file  = h5py.File(input_path)
        in_a     = in_file["main"][:]
    elif input_mode == 'random':
        in_a = np.random.random_sample(in_shape)
    elif input_mode == 'zero':
        in_a = np.zeroes(in_shape)
    elif input_mode == 'one':
        in_a = np.ones(in_shape)


    print "Running net..."
    s = time()
    for i in range(N):
        z.forward(in_a)
    e = time()
    print (e - s) / N


