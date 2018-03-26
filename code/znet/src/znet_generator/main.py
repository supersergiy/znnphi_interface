from generate_znet   import generate_znet
from parse_net       import parse_net
from read_in_weights import read_in_weights
from optimize_net    import optimize_net
import sys

net_path     = sys.argv[1]
weights_path = sys.argv[2]
out_path     = sys.argv[3]
arch         = sys.argv[4]
cores        = sys.argv[5]
ht           = sys.argv[6]
cpu_offset   = sys.argv[7]
opt_flags    = sys.argv[8]

if __name__ == "__main__":
    print "Parsing the network spec..."
    net = parse_net(net_path, arch)
    print "Loading the weights..."
    read_in_weights(net, weights_path)
    print "Optimizing the net..."
    optimize_net(net, opt_flags)
    print "Generating the network..."
    generate_znet(net, out_path, cores, ht, cpu_offset)
    print "Done!"
