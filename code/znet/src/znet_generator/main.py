from generate_znet import generate_znet
from parse_net import parse_net
from read_in_weights import read_in_weights

import sys

net_path     = sys.argv[1]
weights_path = sys.argv[2]
out_path     = sys.argv[3]

SIMD_WIDTH = 8
S = SIMD_WIDTH

if __name__ == "__main__":
    net = parse_net(net_path)
    read_in_weights(net, weights_path)
    generate_znet(net, out_path)
