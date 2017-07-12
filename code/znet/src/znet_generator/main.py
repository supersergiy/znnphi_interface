from generate_znet   import generate_znet, generate_template_znet
from parse_net       import parse_net
from read_in_weights import read_in_weights
from optimize_net    import optimize_net
import sys

net_path     = sys.argv[1]
weights_path = sys.argv[2]
out_path     = sys.argv[3]


SIMD_WIDTH = 8
S = SIMD_WIDTH

if __name__ == "__main__":
    print "Parsing the network spec..."
    net = parse_net(net_path)
    if weights_path == 't': 
        generate_template_znet(net, out_path)
    else:
        print "Loading the weights..."
        read_in_weights(net, weights_path)
        print "Optimizing the net..."
        optimize_net(net)
        print "Generating the network..."
        generate_znet(net, out_path)
        print "Done!"