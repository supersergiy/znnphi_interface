#!/usr/bin/python
from time_net import timing_test
from optparse import OptionParser

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

bases_elu = ["/tests/pni_unet", "/tests/pni_unet_no_bn",
             "/tests/unet_sym_elu", "/tests/unet_sym_bn_elu",
             "/tests/unet_paper_elu", "/tests/unet_paper_bn_elu"]
bases_relu = ["/tests/pni_unet_relu", "/tests/pni_unet_no_bn_relu",
              "/tests/unet_sym", "/tests/unet_sym_bn",
              "/tests/unet_paper", "/tests/unet_papaer_bn"]
opts = ["full_opt", "no_pad", "no_act", "lin", "no_add", "no_opt"]


output_file = file(options.output_path, 'w')

for b in bases_elu:
    core_options = {}
    core_options["conv"] = [options.conv_cores, options.conv_ht]
    core_options["act"]  = [2, 1]
    core_options["lin"]  = [options.lin_cores, options.lin_ht]

    for o in opts:
        result = timing_test(base=b, N=options.num_iter, architecture=options.architecture, core_options=core_options, optimization=o,
                          ignore="nothing", time_each=False, recompile=True, input_mode=options.input_mode, cpu_offset=0, run=True)
        output_file.write("{} {} {}\n".format(b, o, result))
        output_file.flush()

for b in bases_relu:
    core_options = {}
    core_options["conv"] = [options.conv_cores, options.conv_ht]
    core_options["act"]  = [1, 1]
    core_options["lin"]  = [options.lin_cores, options.lin_ht]

    for o in opts:
        result = time_net(base=b, N=options.num_iter, architecture=options.architecture, core_options=core_options, optimization=o,
                          ignore="nothing", time_each=False, recompile=True, input_mode=options.input_mode, cpu_offset=0, run=True)
        output_file.write("{} {} {}\n".format(b, o, result))
        output_file.flush()

