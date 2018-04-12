#!/usr/bin/python3
from time_net import timing_test
from optparse import OptionParser
import os
parser = OptionParser()

parser.add_option("-o", "--output", dest="output_path")
parser.add_option("-i", "--input_mode", dest="input_mode", default="read")
parser.add_option("--iter", dest="num_iter", default=20, type="int")

parser.add_option("-c", "--cores", dest="conv_cores", default=2)
parser.add_option("--ht", dest="conv_ht", default=2)
parser.add_option("--act_cores", dest="act_cores", default=-1)
parser.add_option("--act_ht", dest="act_ht", default=-1)
parser.add_option("--lin_cores", dest="lin_cores", default=-1)
parser.add_option("--lin_ht", dest="lin_ht", default=-1)

parser.add_option("--arch", dest="architecture", default="AVX2",
        help="The cpu architexture: {AVX2, AVX512}")
(options, args) = parser.parse_args()

bases_elu = ["/tests/pni_unet",
             "/tests/unet_sym_bn_elu",
             "/tests/unet_paper_bn_elu"]
bases_relu = [
              "/tests/pni_unet_relu",
              "/tests/unet_sym_bn",
              "/tests/unet_paper_bn"]

opts = ["full_opt", "no_pad", "no_act", "lin", "no_add", "no_opt"]


for b in bases_elu:
    for o in opts:
        run_format  = "./time_net.py -b {} -O {} --act_cores {} --act_ht {} --cores {} --ht {} --recompile --arch {} -o {} --iter {}"
        run_str     = run_format.format(b, o, options.act_cores, options.act_ht, options.conv_cores,
                                        options.conv_ht, options.architecture, options.output_path, options.num_iter)

        os.system(run_str)

for b in bases_relu:
    for o in opts:
        run_format  = "./time_net.py -b {} -O {} --act_cores {} --act_ht {} --cores {} --ht {} --recompile --arch {} -o {} --iter {}"
        run_str     = run_format.format(b, o, options.act_cores, options.act_ht, options.conv_cores,
                                        options.conv_ht, options.architecture, options.output_path, options.num_iter)
        os.system(run_str)
