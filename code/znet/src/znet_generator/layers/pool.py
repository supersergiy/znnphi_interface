import copy
from common import generate_param_string

def set_pool_dim(params, bot_tensor):

    top_dim = copy.copy(bot_tensor.dim)
    top_dim[2] /= params["kernel_dim"][0]
    top_dim[3] /= params["kernel_dim"][1]
    top_dim[4] /= params["kernel_dim"][2]
    params["top_dim"] = top_dim
    params["bot_dim"] = bot_tensor.dim

def parse_pool(json_param, arch):
    params = {}
    params["arch"] = arch
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "pool"

    #TODO: there's a bug with converting pooling param to JSON
    #      will need to un-hardcode this
    params["kernel_dim"] = [1, 2, 2]
    params["stride"]     = [1, 2, 2]


    return params

def allocate_pool_lines(lparam):
    bot_dim = lparam["bot_dim"]
    l = lparam
    pool_params = (bot_dim[0], bot_dim[1], bot_dim[2], bot_dim[3],
                   l["kernel_dim"][0], l["kernel_dim"][1],
                   l["stride"][0], l["stride"][1])
    param_str = generate_param_string(pool_params)
    lines = []
    lines.append('layers["{}"] = new znn::phi::MaxPoolingLayer({});'.format(l["name"],
                                                                            param_str))
    return lines


