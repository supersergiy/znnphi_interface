import copy
from common import round_to_simd, generate_param_string, S, fill_tensor, zero_out_tensor
import numpy as np
from conv import block_kernel, block_bias

def set_bnorm_dim(params, bot_tensor):
    top_dim = copy.copy(bot_tensor.dim)
    params["top_dim"] = top_dim

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = bot_tensor.dim[1]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["scale_size"]  = round_to_simd(params["ifm"])
    params["bias_size"] = round_to_simd(params["ifm"])

def parse_bnorm(json_param):
    params = {}
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "bnorm"
    params["static_bnorm"] = json_param["batch_norm_param"]["use_global_stats"]
    params["scale"] = "scale_{}".format(params["name"])
    params["bias"]  = "bias_{}".format(params["name"])

    return params

def allocate_bnorm_lines(lparam):
    l = lparam
    allocation_params = lparam["top_dim"][0:4]

    param_str = generate_param_string(allocation_params)
    lines = []
    #allocate layer
    if lparam["static_bnorm"]:
       lines.append('layers["{}"] = new znn::phi::ScaleLayer({});'.format(l["name"],
                                                                           param_str))
    else:
       lines.append('layers["{}"] = new znn::phi::DynamicBnormLayer({});'.format(l["name"],
                                                                           param_str))
    #allocate weights
    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                  l["scale"], l["scale_size"]))

    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                  l["bias"], l["bias_size"]))
    #initialize weights
    scale = l["scale_data"]
    lines += fill_tensor('{}'.format(l["scale"]), scale)

    bias = l["bias_data"]
    lines += fill_tensor('{}'.format(l["bias"]), bias)

    return lines

