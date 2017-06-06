import json
import sys
import copy
from math import ceil
from operator import mul
from six import iteritems

import param_parser
from Tensor import Tensor
from codegen import generate_function
#import pdb; pdb.set_trace()

net_path   = "./nets/unet.json"

BATCH_SIZE = 1
SIMD_WIDTH = 8
S = SIMD_WIDTH

def upd_tensor(tensors, name, dim):
    size = reduce(mul, dim)
    if name not in tensors:
        tensors[name] = Tensor(dim)
    elif size > tensors[name].size:
        tensors[name].size = size

def parse_net(net_path):
    with open(net_path) as f:
   net = json.load(f)

    json_layers  = net["layer"]
    tensors      = {}
    layer_order  = []
    layer_info   = {}

    for l in json_layers:
        lparams = {}
   lt = l["type"]
        print lt

        if lt != "Input":
            bot_name   = l["bottom"][0]
            top_name   = l["top"][0]
            bot_tensor = tensors[bot_name]

   if lt == "Input":
       dim = l["input_param"]["shape"][0]["dim"]
       dim[0] = BATCH_SIZE
       dim[1] = int(ceil(dim[1] / S) * S)

            tensors[l["name"]] = Tensor(dim)
   elif lt == "Convolution":
       lparams = param_parser.parse_conv(l["convolution_param"], bot_tensor)
        elif lt == "ELU":
            lparams["top_dim"] = copy.copy(bot_tensor.dim)
        elif lt == "Sigmoid":
            lparams["top_dim"] = copy.copy(bot_tensor.dim)
        elif lt == "BatchNorm":
            lparams["top_dim"] = copy.copy(bot_tensor.dim)
        elif lt == "Scale":
            lparams["top_dim"] = copy.copy(bot_tensor.dim)
        elif lt == "Eltwise":
            lparams["top_dim"] = copy.copy(bot_tensor.dim)
            #TODO: do I need to reset dimensions?
        elif lt == "Pooling":
            lparams = param_parser.parse_pool(l["pooling_param"], bot_tensor)
        elif lt == "Deconvolution":
            lparams = param_parser.parse_deconv(l["convolution_param"], bot_tensor)
        else:
            raise Exception("Unsupported Layer: {}".format(lt))

        if lparams:
            if "top_dim" in lparams:
                top_dim = lparams["top_dim"]
                upd_tensor(tensors, top_name, top_dim)

            layer_info[l["name"]] = lparams
            layer_order.append(l["name"])

    return (tensors, layer_info, layer_order)

def generate_constructor_body(net):
    lines = []
    tensors, layer_info, _ = net
    #alocate tensors
    for (n,t) in iteritems(tensors):
       lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(n, t.size))
   
    #allocate weights
    for (n,t) in iteritems(tensors):
       lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(n, t.size))
    #initialize weights
    #allocate layers
    #initialize layers
    return lines

def generate_forward_body(net):
    lines = []
    return lines

def generate_znet(net):
    lines = []
    #includes
    lines.append('#include <iostream>')
    lines.append('#include <chrono>')
    lines.append('#include <znn/interface/conv_wrapper.hpp>')
    lines.append('#include "znet.hpp"')
    lines.append('')

    #initialization
    constructor_signature = 'znn::phi::Znet::Znet(void)'
    constructor_body      = generate_constructor_body(net)
    constructor           = generate_function(constructor_signature, constructor_body)
    lines += constructor

    #forward pass
    forward_signature = 'void znn::phi::Znet::forward(void)'
    forward_body      = generate_forward_body(net)
    forward           = generate_function(forward_signature, forward_body)
    lines += forward

    #write lines to file
    f = open("cpp_out/znet.cpp", 'w')
    for l in lines:
        f.write("{}\n".format(l))

if __name__ == "__main__":
    net = parse_net(net_path)
    generate_znet(net)
