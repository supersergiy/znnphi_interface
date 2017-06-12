import copy
from common import round_to_simd

def get_deconv_top_dim(params, bot_tensor):
    top_dim = [-1, -1, -1, -1, -1]
    top_dim[0] = bot_tensor.dim[0]
    top_dim[1] = params["ofm"]

    for i in [2, 3, 4]:
	top_dim[i]  = params["stride"][i - 2] * (bot_tensor.dim[i] - 1)
        top_dim[i] += params["kernel_dim"][i] - 2*params["pad"][i - 2]
    return top_dim


def get_conv_top_dim(params, bot_tensor):
    top_dim = [-1, -1, -1, -1, -1]
    top_dim[0] = bot_tensor.dim[0]
    top_dim[1] = params["ofm"]
    for i in [2, 3, 4]:
	top_dim[i] = (bot_tensor.dim[i] - params["kernel_dim"][i] +
		                         2*params["pad"][i - 2]) / params["stride"][i - 2] + 1
    return top_dim

def parse_conv(json_conv_param, bot_tensor):
    params = {}
    params["type"] = "conv"

    params["pad"]  = json_conv_param["pad"]
    params["stride"]  = json_conv_param["stride"]

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = json_conv_param["num_output"]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["kernel_dim"]  = [params["ofm"], params["ifm"]]
    params["kernel_dim"] += json_conv_param["kernel_size"]

    params["kernel_size"]  = params["kernel_dim"][2] * params["kernel_dim"][3] * params["kernel_dim"][4]
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][0])
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][1])

    params["bias_dim"] = [params["ofm"]]
    params["bias_size"] = round_to_simd(params["bias_dim"][0])

    params["top_dim"] = get_conv_top_dim(params, bot_tensor)
    return params

def parse_deconv(json_conv_param, bot_tensor):
    params = {}
    params["type"] = "deconv"

    params["pad"]     = json_conv_param["pad"]
    params["stride"]  = json_conv_param["stride"]

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = json_conv_param["num_output"]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["kernel_dim"]  = [params["ofm"], params["ifm"]]
    params["kernel_dim"] += json_conv_param["kernel_size"]


    params["kernel_size"]  = params["kernel_dim"][2] * params["kernel_dim"][3] * params["kernel_dim"][4]
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][0])
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][1])

    params["bias_dim"] = [params["ofm"]]
    params["bias_size"] = round_to_simd(params["bias_dim"][0])

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
