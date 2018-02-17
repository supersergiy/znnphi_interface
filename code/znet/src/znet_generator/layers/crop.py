import copy
from common import round_to_simd, generate_param_string, fill_tensor, zero_out_tensor
import numpy as np
from conv import block_kernel, block_bias

def set_crop_dim(params, bot_tensors):
    top_dim = copy.copy(bot_tensors[1].dim)
    bot_dim = copy.copy(bot_tensors[0].dim)

    params["top_dim"] = top_dim

    params["bn"]  = bot_dim[0]
    params["ifm"] = bot_dim[1]
    params["ofm"] = bot_dim[1]
    params["id"]  = bot_dim[2]
    params["ihw"] = bot_dim[3]
    params["od"]  = top_dim[2]
    params["ohw"] = top_dim[3]


def parse_crop(json_param, arch):
    params = {}
    params["arch"] = "AVX2"
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"]
    params["type"] = "crop"

    if json_param["crop_param"]["axis"] != 2:
        raise Exception("Non-spacial cropping not implemented (crop axis != 2)")

    params["z_offset"]   = json_param["crop_param"]["offset"][0]
    params["xy_offset"]  = json_param["crop_param"]["offset"][1]

    return params

def allocate_crop_lines(lparam):
    l = lparam
    allocation_params = [l["bn"], l["ifm"], l["id"], l["ihw"], l["ofm"], l["od"], l["ohw"], l["z_offset"], l["xy_offset"]]

    param_str = generate_param_string(allocation_params)
    lines = []
    #allocate layer
    lines.append('layers["{}"] = new znn::phi::CropLayer({});'.format(l["name"],
                                                                        param_str))
    #allocate weights
    return lines

