from math import ceil
import copy

#TODO: FIX FOR VARIABLE SIMD
S = 8
def round_to_simd(n):
    return ceil(n / S) * S;

def get_deconv_top_dim(params, bot_tensor):
    top_dim = [-1, -1, -1, -1, -1]
    top_dim[0] = bot_tensor.dim[0]
    top_dim[1] = round_to_simd(params["ofm"])

    for i in [2, 3, 4]:
	top_dim[i] = (bot_tensor.dim[i] - params["kdim"][i - 2] +
		                         2*params["pad"][i - 2]) / params["stride"][i - 2] + 1
    return top_dim


def get_conv_top_dim(params, bot_tensor):
    top_dim = [-1, -1, -1, -1, -1]
    top_dim[0] = bot_tensor.dim[0]
    top_dim[1] = round_to_simd(params["ofm"])

    for i in [2, 3, 4]:
	top_dim[i] = (bot_tensor.dim[i] - params["kdim"][i - 2] +
		                         2*params["pad"][i - 2]) / params["stride"][i - 2] + 1
    return top_dim

def parse_conv(json_param, bot_tensor):
    params = {}
    params["type"] = "conv"
    params["kdim"] = json_param["kernel_size"]
    params["pad"]  = json_param["pad"]

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = json_param["num_output"]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["top_dim"] = get_conv_top_dim(params, bot_tensor)

    return params

def parse_deconv(json_param, bot_tensor):
    params = {}
    params["type"] = "deconv"

    params["kdim"] = json_param["kernel_size"]
    params["pad"]  = json_param["pad"]

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = json_param["num_output"]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["top_dim"] = get_deconv_top_dim(params, bot_tensor)

    return params

def parse_pool(json_param, bot_tensor):
    params = {}
    params["type"] = "pool"

    #TODO: there's a bug with converting pooling param to JSON
    #      will need to un-hardcode this
    params["top_dim"]     = copy.copy(bot_tensor.dim)
    params["top_dim"][3] /= 2
    params["top_dim"][4] /= 2

    return params
