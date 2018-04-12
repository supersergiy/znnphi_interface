import copy
from .common import round_to_simd, generate_param_string, fill_tensor, zero_out_tensor
import numpy as np

def set_bias_dim(params, bot_tensor):
    top_dim = copy.copy(bot_tensor.dim)
    params["top_dim"] = top_dim

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = bot_tensor.dim[1]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["scale_size"]  = round_to_simd(params["ifm"], params["arch"])
    params["bias_size"] = round_to_simd(params["ifm"], params["arch"])

def parse_bias(json_param, arch):
    params = {}
    params["arch"] = arch
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "bias"

    params["scale"] = "scale_{}".format(params["name"])
    params["bias"]  = "bias_{}".format(params["name"])

    return params

# all biases must be converted to scales prior to allocation
#def allocate_bias_lines(lparam):

