import copy
from common import round_to_simd, generate_param_string, S, fill_tensor, zero_out_tensor
from conv   import block_kernel, block_bias

def set_deconv_dim(params, bot_tensor):
    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["kernel_dim"]  = [params["ofm"], params["ifm"]]
    params["kernel_dim"] += params["json_kernel_size"]

    params["kernel_size"]  = params["kernel_dim"][2] * params["kernel_dim"][3] * params["kernel_dim"][4]
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][0])
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][1])

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

def allocate_deconv_lines(lparam):
    l = lparam

    allocation_params = (l["bn"], l["ifm"], l["ofm"], l["id"], l["ihw"],
                         l["kernel_dim"][2], l["kernel_dim"][3],
                         l["stride"][0],  l["stride"][1])

    param_str = generate_param_string(allocation_params)
    lines = []
    #allocate layer
    lines.append('layers["{}"] = new znn::phi::DeconvLayer({});'.format(l["name"],
                                                                        param_str))
    #allocate weights 
    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                              l["kernel"], l["kernel_size"]))

    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                  l["bias"], l["bias_size"]))
    #initialize weights
    kernel = l["kernel_data"] 
    blocked_kernel = block_kernel(kernel, l)
    lines += fill_tensor('{}_kernel'.format(l["name"]), blocked_kernel)

    bias = l["bias_data"] 

    if bias is None: 
        lines += zero_out_tensor('{}_bias'.format(l["name"])) #TODO: don't actually have to allocate all tensors, but then have to allocate one biggest one
    else:
        blocked_bias = block_bias(bias, l)
        lines += fill_tensor('{}_bias'.format(l["name"]), blocked_bias)

    return lines

