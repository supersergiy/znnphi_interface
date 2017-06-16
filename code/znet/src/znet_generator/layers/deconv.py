import copy
from common import round_to_simd

def set_deconv_dim(params, bot_tensor):
    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["kernel_dim"]  = [params["ofm"], params["ifm"]]
    params["kernel_dim"] += params["json_kernel_size"]

    params["kernel_size"]  = params["kernel_dim"][2] * params["kernel_dim"][3] * params["kernel_dim"][4]
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][0])
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][0])

    top_dim = [-1, -1, -1, -1, -1]
    top_dim[0] = bot_tensor.dim[0]
    top_dim[1] = params["ofm"]
    for i in [2, 3, 4]:
	top_dim[i]  = params["stride"][i - 2] * (bot_tensor.dim[i] - 1)
        top_dim[i] += params["kernel_dim"][i] - 2*params["pad"][i - 2]
    params["top_dim"] = top_dim

def parse_deconv(json_param):
    params = {}
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "deconv"

    json_conv_param = json_param["convolution_param"]

    if "pad" in json_conv_param:
        params["pad"]     = json_conv_param["pad"]
    else:
        params["pad"]     = [0, 0, 0]
    params["stride"]  = json_conv_param["stride"]

    params["ofm"] = json_conv_param["num_output"]

    params["json_kernel_size"] = json_conv_param["kernel_size"]
    params["kernel"] = "{}_kernel".format(params["name"])
    params["bias"]   = "{}_bias".format(params["name"])

    params["bias_dim"] = [params["ofm"]]
    params["bias_size"] = round_to_simd(params["bias_dim"][0])

    return params

