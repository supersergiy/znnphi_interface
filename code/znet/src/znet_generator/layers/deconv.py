import copy
from common import round_to_simd, generate_param_string, S, fill_tensor, zero_out_tensor
import numpy as np

def set_deconv_dim(params, bot_tensor):
    params["bn"]  = bot_tensor.dim[0]
    params["ifm"] = bot_tensor.dim[1]
    params["id"]  = bot_tensor.dim[2]
    params["ihw"] = bot_tensor.dim[3]

    params["kernel_dim"]  = [params["ifm"], params["ofm"]]
    params["kernel_dim"] += params["json_kernel_size"]

    params["kernel_size"]  = params["kernel_dim"][2] * params["kernel_dim"][3] * params["kernel_dim"][4]
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][0])
    params["kernel_size"] *= round_to_simd(params["kernel_dim"][1])
    params["kernel_size"] /= params["group"]

    top_dim = [-1, -1, -1, -1, -1]
    top_dim[0] = bot_tensor.dim[0]
    top_dim[1] = params["ofm"]
    for i in [2, 3, 4]:
	top_dim[i]  = params["stride"][i - 2] * (bot_tensor.dim[i] - 1)
        top_dim[i] += params["kernel_dim"][i] - 2*params["pad"][i - 2]

    params["top_dim"] = top_dim
    params["bot_dim"] = bot_tensor.dim

def parse_deconv(json_param):
    params = {}
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "deconv"

    json_conv_param = json_param["convolution_param"]
    params["additive_conv"] = False

    if "pad" in json_conv_param:
        params["pad"] = json_conv_param["pad"]
    else:
        params["pad"] = [0, 0, 0]

    if "group" in json_conv_param:
        params["group"] = json_conv_param["group"]
    else:
        params["group"] = 1

    params["stride"]  = json_conv_param["stride"]

    params["ofm"] = json_conv_param["num_output"]

    params["json_kernel_size"] = json_conv_param["kernel_size"]
    params["kernel"] = "{}_kernel".format(params["name"])
    params["bias"]   = "{}_bias".format(params["name"])

    params["bias_dim"] = [params["ofm"]]
    params["bias_size"] = round_to_simd(params["bias_dim"][0])
    params["scale_size"] = round_to_simd(params["ofm"])

    params["scale"] = "{}_scale".format(params["name"])

    return params

def block_kernel(kernel, lparam):
    kdim = lparam["kernel_dim"]
    kernel = kernel.reshape(kdim)
    blocked_kernel = np.array([0.0]*lparam['kernel_size'])
    def h5ker_to_znnphiker(ifm, ofm, kz, kx, ky):
        total_ifms = round_to_simd(kdim[0])
        total_ofms = round_to_simd(kdim[1])

        offset = ofm/S
        offset *= total_ifms/S
        offset += ifm/S
        offset *= kdim[2]
        offset += kz
        offset *= kdim[3]
        offset += kx
        offset *= kdim[4]
        offset += ky
        offset *= S
        offset += ifm % S
        offset *= S
        offset += ofm % S
        return offset

    # h5 weight format: ifm-ofm-kz-kx-ky
    # output format: ofm/S-ifm/S-kz-kx-ky-ifm%S-ofm%S

    for ifm in range(kdim[0]):
        for ofm in range(kdim[1]):
            for kz in range(kdim[2]):
                for kx in range(kdim[3]):
                    for ky in range(kdim[4]):
                        znnphi_index = h5ker_to_znnphiker(ifm, ofm, kz, kx, ky)
                        blocked_kernel[znnphi_index] = kernel[ifm][ofm][kz][kx][ky]
    return blocked_kernel


def allocate_deconv_lines(lparam):
    l = lparam
    if "activation" in l and l["activation"] == "elu":
        activate = "true"
    else:
        activate = "false"

    if "additive_conv" in l and l["additive_conv"] == True:
        add_or_overwrite = "true"
    else:
        add_or_overwrite = "false"

    allocation_params = (l["bn"], l["ifm"], l["ofm"], l["id"], l["ihw"],
                         l["kernel_dim"][2], l["kernel_dim"][3],
                         l["stride"][0],  l["stride"][1],
                         0, 0, activate, add_or_overwrite,
                         'tensors["{}"]->data()'.format(l["kernel"]),
                         l["cores"], l["ht"], l["cpu_offset"])
              #           #'tensors["{}"]->data()'.format(l["bias"]))

    param_str = generate_param_string(allocation_params)

    if lparam["group"] == lparam["ofm"] and lparam["ifm"] == lparam["ofm"]:
        return allocate_deconv_as_interpolation(lparam, param_str)
    elif lparam["group"] == 1:
        return allocate_deconv_as_conv(lparam, param_str)
    else:
        raise Exception("No suitable implementation for {}".format(lparam["name"]))

def allocate_deconv_as_interpolation(lparam, param_str):
    l = lparam
    lines = []

    #allocate weights
    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                              l["kernel"], l["kernel_size"]))

    #initialize weights
    kernel = l["kernel_data"]
    lines += fill_tensor(l["kernel"], kernel.flatten())

    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                              l["bias"], l["bias_size"]))
    bias = l["bias_data"]
    assert( bias is None or np.sum(np.abs(bias)) == 0 )
    lines += zero_out_tensor(l["bias"])

    lines.append('layers["{}"] = new znn::phi::InterpolationLayer({});'.format(l["name"],
                                                                               param_str))
    #lines.append('layers["{}"] = new znn::phi::DummyLayer();')
    return lines

def allocate_deconv_as_conv(lparam, param_str):
    l = lparam

    lines = []
    #allocate weights
    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                              l["kernel"], l["kernel_size"]))

    lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                  l["bias"], l["bias_size"]))
    #initialize weights
    kernel = l["kernel_data"]
    blocked_kernel = block_kernel(kernel, l)
    lines += fill_tensor('{}_kernel'.format(l["name"]), blocked_kernel.flatten())

    bias = l["bias_data"]
    if bias is None:
        lines += zero_out_tensor('{}_bias'.format(l["name"])) #TODO: don't actually have to allocate all tensors, but then have to allocate one biggest one
    else:
        lines += fill_tensor('{}_bias'.format(l["name"]), bias.flatten() )

    #allocate layer
    #ouch, this is a little ugly with the lib path, but sacrafices have to made to keep it dynamic
    lines.append('layers["{}"] = new znn::phi::DeconvAsConvLayer({}, lib_path=this->lib_path);'.format(l["name"], param_str))
    if "additive_conv" in l and l["additive_conv"]:
        lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                  l["scale"], l["scale_size"]))
        lines += fill_tensor('{}'.format(l["scale"]), l["scale_data"])

    return lines

