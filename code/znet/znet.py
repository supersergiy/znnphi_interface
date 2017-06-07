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
      name = l["name"]
      lt = l["type"]
      print lt
      lparams = {}

      if lt != "Input":
         bot_name   = l["bottom"][0]
         top_name   = l["top"][0]
         bot_tensor = tensors[bot_name]

      if lt == "Input":
         lparams["type"] = "input"
         dim = l["input_param"]["shape"][0]["dim"]
         dim[0] = BATCH_SIZE
         dim[1] = int(ceil(dim[1] / S) * S)
         tensors[l["name"]] = Tensor(dim)
      elif lt == "Convolution":
         lparams = param_parser.parse_conv(l["convolution_param"], bot_tensor)
      elif lt == "Deconvolution":
         lparams = param_parser.parse_deconv(l["convolution_param"], bot_tensor)
      elif lt == "Pooling":
         lparams["type"] = "pool"
         lparams = param_parser.parse_pool(l["pooling_param"], bot_tensor)
      elif lt == "ELU":
         lparams["type"] = "elu"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
      elif lt == "Sigmoid":
         lparams["type"] = "sigmoid"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
      elif lt == "BatchNorm":
         lparams["type"] = "batchnorm"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
      elif lt == "Scale":
         lparams["type"] = "scale"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
      elif lt == "Eltwise":
         lparams["type"] = "eltwise"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
          #TODO: do I need to reset dimensions here?
      else:
         raise Exception("Unsupported Layer: {}".format(lt))

      lparams["name"] = name
      if "top_dim" in lparams:
         top_dim = lparams["top_dim"]
         upd_tensor(tensors, top_name, top_dim)

      if lparams["type"] != "input":
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
    for (n,l) in iteritems(layer_info):
       if l["type"] in ["conv", "deconv"]:
          lines.append('weights["{}_kernel"] = new znn::phi::hbw_array<float>({});'.format(n,
                      l["kernel_size"]))
          lines.append('weights["{}_bias"] = new znn::phi::hbw_array<float>({});'.format(n,
                      l["bias_size"]))
    lines.append('')
    #initialize weights
    #TODO: read weights from HD5, convert them to the right layout, hardcode them in
    lines.append('')

    #allocate layers
    if l["type"] == "conv":
       lines.append(
          'layers["{}"] = new znn::phi::ConvWrapper();'.format(l["name"]))

    lines.append('')

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
