import json
import sys
import copy
from operator import mul
from six import iteritems
import h5py
import numpy as np
from collections import deque

import param_parser
from Tensor import Tensor
from generate import generate_znet

net_path     = sys.argv[1]
weights_path = sys.argv[2]
#weights_path = None

BATCH_SIZE =1
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
   layer_order  = deque()
   layer_info   = {}

   for l in json_layers:
      name = l["name"]
      lt = l["type"]
      lparams = {}

      if lt != "Input":
         bot_name   = l["bottom"][0]
         top_name   = l["top"][0]
         bot_tensor = tensors[bot_name]

      if lt == "Input":
         lparams["type"] = "input"
         dim = l["input_param"]["shape"][0]["dim"]
         dim[0] = BATCH_SIZE
         tensors[l["name"]] = Tensor(dim)
      elif lt == "Convolution":
         lparams = param_parser.parse_conv(l["convolution_param"], bot_tensor)
         lparams["kernel"] = "{}_kernel".format(name)
         lparams["bias"]   = "{}_bias".format(name)
      elif lt == "Deconvolution":
         lparams = param_parser.parse_deconv(l["convolution_param"], bot_tensor)
         lparams["kernel"] = "{}_kernel".format(name)
         lparams["bias"]   = "{}_bias".format(name)
      elif lt == "Pooling":
         lparams["type"] = "pool"
         lparams = param_parser.parse_pool(l["pooling_param"], bot_tensor)
      elif lt == "ELU":
         lparams["type"] = "elu"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
      elif lt == "Sigmoid":
         lparams["type"] = "sigmoid"
         lparams["top_dim"] = copy.copy(bot_tensor.dim)
         lparams["top"] = l["top"]
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
         lparams["top"] = top_name
         lparams["bot"] = bot_name
         layer_info[l["name"]] = lparams
         layer_order.append(l["name"])
   return (tensors, layer_info, layer_order)

def add_block_input(net):
    tensors, layer_info, layer_order = net

    block_params = {
                        "type": "block_input",
                        "name": "block_input",
                        "bot": "user_input",
                        "top": "input"
                   }
    layer_order.appendleft("block_input")
    layer_info["block_input"] = block_params
    tensors["user_input"] = Tensor(tensors["input"].dim)

def add_unblock_output(net):
    tensors, layer_info, layer_order = net
    if "output" in tensors:
        block_params = {}
        block_params = {
                            "type": "unblock_output",
                            "name": "unblock_output",
                            "bot": "output",
                            "top": "user_output"
                       }
        layer_order.append("unblock_output")
        layer_info["unblock_output"] = block_params
        tensors["user_output"] = Tensor(tensors["output"].dim)


if __name__ == "__main__":
    net = parse_net(net_path)
    add_block_input(net)
    add_unblock_output(net)
    generate_znet(net, weights_path)
