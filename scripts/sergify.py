#!/usr/bin/python3
import os
import numpy as np
import h5py
import sys
from optparse import OptionParser

from pznet.pznet import PZNet

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
parser.add_option("--act-cores", dest="act_cores", default=-1)
parser.add_option("--act-ht", dest="act_ht", default=-1)
parser.add_option("--lin-cores", dest="lin_cores", default=-1)
parser.add_option("--lin-ht", dest="lin_ht", default=-1)
parser.add_option("--cpu-offset", dest="cpu_offset", default=0)
parser.add_option("-o", "--output-znet-path", dest="output_znet_path", 
                  default="./generated_netwrok",
                  help="The output folder for the generated network")

parser.add_option("--arch", dest="architecture", default="AVX2",
        help="The cpu architexture: {AVX2, AVX512}")
(options, args) = parser.parse_args()

if not options.prototxt_path:
    parser.error("Network prototxt path not given")
if not options.weights_path:
    parser.error("Weights path not given")
output_znet_path = os.path.abspath(options.output_znet_path)

core_options = {}
core_options["conv"] = [options.conv_cores, options.conv_ht]
core_options["act"]  = [options.act_cores, options.act_ht]
core_options["lin"]  = [options.lin_cores, options.lin_ht]

print ("Creating the network...")
net = PZNet.from_kaffe_model(
    options.prototxt_path, 
    options.weights_path, 
    output_znet_path,
    architecture=options.architecture, 
    core_options=core_options,
    cpu_offset=options.cpu_offset,
    opt_mode='full_opt',
    ignore='',
    time_each=options.time_layers
)

print ("Compiling layers...")
breakpoint()
PZNet(output_znet_path)
print ("Your network has been sergified! You can find it at {}".format(output_znet_path))
