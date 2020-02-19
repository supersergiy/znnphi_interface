import copy
from .common import generate_param_string

def set_eltwise_dim(params, bot_tensors):
    bot_tensor = bot_tensors[0]
    top_dim = copy.copy(bot_tensor.dim)
    params["top_dim"] = top_dim

    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["ofm"] = bot_tensor.dim[1]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

def parse_eltwise(json_param, arch):
    params = {}
    params["arch"] = arch
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"] # Expecting 2 bots
    params["type"] = "eltwise"
    if "EltwiseParameter" in json_param and "EltwiseOp" in json_param["EltwiseParameter"]:
        params["mode"] = json_param["EltwiseParameter"]["EltwiseOp"]
    else:
        params["mode"] = 1 # Default (SUM)

    return params

def allocate_eltwise_lines(lparam):
    l = lparam
    allocation_params = l["top_dim"][0:4]  + [l["mode"]]

    param_str = generate_param_string(allocation_params)
    lines = []
    #allocate layer
    lines.append('layers["{}"] = new znn::phi::EltwiseLayer({});'.format(l["name"],
                                                                        param_str))
    return lines

