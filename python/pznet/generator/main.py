from generate_znet   import generate_znet
from parse_net       import parse_net
from read_in_weights import read_in_weights
from optimize_net    import optimize_net
import sys
from optparse import OptionParser

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = OptionParser()

parser.add_option("-n", "--net-path", dest="net_path")
parser.add_option("-w", "--weights-path", dest="weights_path")
parser.add_option("-o", "--out-path", dest="out_path")
parser.add_option("-i", "--input-mode", dest="input_mode", default="read")
parser.add_option("-O", dest="opt_mode", default="full_opt")
parser.add_option("--cpu-offset", dest="cpu_offset", default=0)
parser.add_option("--conv-cores", dest="conv_cores")
parser.add_option("--conv-ht", dest="conv_ht")
parser.add_option("--act-cores", dest="act_cores")
parser.add_option("--act-ht", dest="act_ht")
parser.add_option("--lin-cores", dest="lin_cores")
parser.add_option("--lin-ht", dest="lin_ht")
parser.add_option("--architecture", dest="architecture", default="AVX2",
        help="The cpu architexture: {AVX2, AVX512}")
parser.add_option("--ignore", dest="ignore", default="")
parser.add_option("--time_each", dest="time_each", default="False")

(options, args) = parser.parse_args()

net_path     = options.net_path
weights_path = options.weights_path
out_path     = options.out_path
arch         = options.architecture
cpu_offset   = options.cpu_offset
opt_mode     = options.opt_mode
ignore       = options.ignore
time_each    = str2bool(options.time_each)

core_options = {
                "conv": (options.conv_cores, options.conv_ht),
                "deconv": (options.conv_cores, options.conv_ht),
                "elu": (options.act_cores, options.act_ht),#translating to layer type here TODO
                "relu": (options.act_cores, options.act_ht),
                "scale": (options.lin_cores, options.lin_ht),
                "bnorm": (options.lin_cores, options.lin_ht)
               }

if __name__ == "__main__":
    print ("Parsing the network spec...")
    net = parse_net(net_path, arch)
    print ("Loading the weights...")
    read_in_weights(net, weights_path)
    print ("Optimizing the net...")
    optimize_net(net, opt_mode)
    print ("Generating the network...")
    generate_znet(net, out_path, core_options, cpu_offset, ignore, time_each)
    print ("Done!")
