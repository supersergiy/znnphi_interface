#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-n", "--net", dest="prototxt_path",
                          help="path to the caffe .prototxt file", metavar="FILE")
parser.add_option("-w", "--weights", dest="weights_path",
                          help="path to the weights .h5 file", metavar="FILE")
parser.add_option("-t", "--timelayers",
                          action="store_false", dest="time_layers", default=False,
                          help="Time each layer separately")
parser.add_option("-c", "--cores", dest="conv_cores", default=2)
parser.add_option("--ht", dest="conv_ht", default=2)
parser.add_option("--act_cores", dest="act_cores", default=-1)
parser.add_option("--act_ht", dest="act_ht", default=-1)
parser.add_option("--lin_cores", dest="lin_cores", default=-1)
parser.add_option("--lin_ht", dest="lin_ht", default=-1)
parser.add_option("-o", dest="output_path", default="./generated_netwrok",
                        help="The output folder for the generated network")

parser.add_option("--arch", dest="architecture", default="AVX2",
        help="The cpu architexture: {AVX2, AVX512}")
(options, args) = parser.parse_args()

if not options.prototxt_path:
    parser.error("Network prototxt path not given")
if not options.weights_path:
    parser.error("Weights path not given")

weights_path = options.weights_path
net_path  = options.prototxt_path
znet_path = os.path.abspath(options.output_path)
arch	  = options.architecture

core_options = {}
core_options["conv"] = [options.conv_cores, options.conv_ht]
core_options["act"]  = [options.act_cores, options.act_ht]
core_options["lin"]  = [options.lin_cores, options.lin_ht]

z = pznet.znet()
print "Creating the network..."
z.create_net(net_path, weights_path, znet_path, arch, core_options=core_options)
print "Compiling layers..."
z.load_net(znet_path, os.path.join(znet_path, "lib"))
print "Your network has been sergified! You can find it at {}".format(znet_path)
