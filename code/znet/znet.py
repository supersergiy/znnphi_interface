import json
from math import ceil
from Tensor import Tensor
from operator import mul
import param_parser

net_path   = "./nets/unet.json"
BATCH_SIZE = 1
SIMD_WIDTH = 8
S = SIMD_WIDTH

def upd_tensor(tensors, name, dim):
    size = reduce(mul, list)
    if name not in tensors:
        tensors[name] = Tensor(size)
    elif size > tensors[name].size:
        tensors[name].size = size

def parse_net(net_path, layers, tensors):
    with open(net_path) as f:
	net = json.load(f)

    json_layers  = net["layer"]
    tensors      = {}
    layer_order  = []
    layer_info   = {}

    for l in json_layers:
	lt = l["type"]
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
            layer_info[l["name"]] = lparams
            layer_order.push(l["name"])
        elif lt == "ELU":
            pass
        elif lt == "BatchNorm":
            pass
        elif lt == "Scale":
            pass
        elif lt == "Eltwise":
            #TODO: do I need to reset dimensions?
            pass
        elif lt == "Pooling":
	    lparams = param_parser.parse_pool(l["convolution_param"], bot_tensor)
            layer_info[l["name"]] = lparams
            layer_order.push(l["name"])
        elif lt == "Deconvolution":
	    lparams = param_parser.parse_deconv(l["convolution_param"], bot_tensor)
            layer_info[l["name"]] = lparams
            layer_order.push(l["name"])
        else:
            raise Exception("Unsupported Layer: {}".format(lt))

        if "top_dim" in lparams:
            top_dim = lparams["top_dim"]
            upd_tensor(tensors, top_name, top_dim)



