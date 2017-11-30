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
parser.add_option("-c", "--cores",
                          dest="n_cores", default=2,
                          help="Number of cores to use")
parser.add_option("--ht",
                          dest="n_ht", default=2,
                          help="Number of hyperthread per core")
parser.add_option("-o", dest="output_path", default="./generated_netwrok",
                        help="The output folder for the generated network")

(options, args) = parser.parse_args()

if not options.prototxt_path:
    parser.error("Network prototxt path not given")
if not options.weights_path:
    parser.error("Weights path not given")

weights_path = options.weights_path
net_path  = options.prototxt_path
znet_path = options.output_path
n_cores   = options.n_cores
n_h5      = options.n_ht

z = pznet.znet()
print "Creating the network..."
z.create_net(net_path, weights_path, znet_path, n_cores, n_ht)
print "Compiling layers..."
z.load_net(znet_path, os.path.join(znet_path, "lib"))
print "Your network has been sergified! You can find it at {}".format(znet_path)
