import copy
from .common import round_to_simd, generate_param_string, fill_tensor, zero_out_tensor
import numpy as np
from .conv import block_kernel, block_bias

def set_slc_dim(params, bot_tensor):
    top_dim = copy.copy(bot_tensor.dim)
    top_dim[1] = params["slice_point"]

    params["top_dim"] = top_dim
    params["bot_dim"] = bot_tensor.dim

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = params["slice_point"]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]


def parse_slc(json_param, arch):
    params = {}
    params["arch"] = arch
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "slc"
    params["slice_point"] = int(json_param["slice_param"]["slice_point"][0])
    return params

def allocate_slc_lines(lparam):
    l = lparam
    allocation_params = lparam["bot_dim"][:4] + [lparam["slice_point"]]
    param_str = generate_param_string(allocation_params)
    lines = []
    #allocate layer
    lines.append('layers["{}"] = new znn::phi::SliceLayer({});'.format(l["name"],
                                                                        param_str))
    return lines

