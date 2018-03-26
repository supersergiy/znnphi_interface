import copy
from common import round_to_simd, generate_param_string, fill_tensor, zero_out_tensor
import numpy as np
from conv import block_kernel, block_bias

def set_mergecrop_dim(params, bot_tensors):
    top_dim = copy.copy(bot_tensors[1].dim)
    bot_dim = copy.copy(bot_tensors[0].dim)

    params["top_dim"] = top_dim

    params["bn"]  = top_dim[0]
    params["ifm1"] = bot_tensors[0][1]
    params["id1"]  = bot_tensors[0][2]
    params["ihw1"] = bot_tensors[0][3]
    params["ifm"]  = bot_tensors[0][1]
    params["id"]   = bot_tensors[0][2]
    params["ihw"]  = bot_tensors[0][3]
    params["ifm2"] = bot_tensors[1][1]
    params["id2"]  = bot_tensors[1][2]
    params["ihw2"] = bot_tensors[1][3]

    params["ofm"] = params["ifm1"] + params["ifm2"]
    params["od"]  = params["id2"]
    params["ohw"] = params["ihw2"]

    params["z_offset"]  = (params["id2"]  - params["id1"] ) / 2
    params["xy_offset"] = (params["ihw2"] - params["ihw1"]) / 2


def parse_mergecrop(json_param, arch):
    params = {}
    params["arch"] = arch
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"]
    params["type"] = "mergecrop"

    return params

def allocate_mergecrop_lines(lparam):
    raise Exception("Mergecrop layer should be broken into crop and merge")

