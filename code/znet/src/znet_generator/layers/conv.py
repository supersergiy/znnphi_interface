import copy
from common import round_to_simd, generate_param_string, S, fill_tensor, zero_out_tensor
import numpy as np

def set_conv_dim(params, bot_tensor):
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
	top_dim[i] = (bot_tensor.dim[i] - params["kernel_dim"][i] +
		                         2*params["pad"][i - 2]) / params["stride"][i - 2] + 1
    params["top_dim"] = top_dim
    params["bot_dim"] = bot_tensor.dim
    params["bot_size"] = bot_tensor.memory_size

def parse_conv(json_param):
    params = {}
    params["name"] = json_param["name"]
    params["top"]  = json_param["top"][0]
    params["bot"]  = json_param["bottom"][0]
    params["type"] = "conv"

    json_conv_param = json_param["convolution_param"]

    if "pad" in json_conv_param:
        params["pad"]     = json_conv_param["pad"]
    else:
        params["pad"]     = [0, 0, 0]
    params["out_pad"]     = [0, 0, 0]

    params["stride"]  = json_conv_param["stride"]

    params["ofm"] = json_conv_param["num_output"]

    params["json_kernel_size"] = json_conv_param["kernel_size"]
    params["kernel"] = "{}_kernel".format(params["name"])
    params["bias"]   = "{}_bias".format(params["name"])

    params["bias_dim"] = [params["ofm"]]
    params["bias_size"] = round_to_simd(params["bias_dim"][0])

    params["activation"]    = None

    params["additive_conv"] = False
    params["scale"] = "{}_scale".format(params["name"])
    params["scale_size"] = round_to_simd(params["ofm"])
    params["scale_data"] = None
    return params

def block_kernel(kernel, lparam):
    kdim = lparam["kernel_dim"]
    kernel = kernel.reshape(kdim)
    blocked_kernel = np.array([0.0]*lparam['kernel_size'])

    def h5ker_to_znnphiker(ofm, ifm, kz, kx, ky):
        total_ofms = round_to_simd(kdim[0])
        total_ifms = round_to_simd(kdim[1])

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

    # h5 weight format: ofm-ifm-kz-kx-ky
    # output format: ofm/S-ifm/S-kz-kx-ky-ofm%S-ifm%S
    for ofm in range(kdim[0]):
        for ifm in range(kdim[1]):
            for kz in range(kdim[2]):
                for kx in range(kdim[3]):
                    for ky in range(kdim[4]):
                        znnphi_index = h5ker_to_znnphiker(ofm, ifm, kz, kx, ky)
                        blocked_kernel[znnphi_index] = kernel[ofm][ifm][kz][kx][ky]

    return blocked_kernel


def block_bias(bias, lparam):
    blocked_bias = np.array([0.0]*lparam['bias_size'])
    for ofm in range(lparam["ofm"]):
        blocked_bias[ofm] = bias[ofm]

    return blocked_bias

def allocate_conv_lines(lparam):
    l = lparam
    if l["pad"][0] != 0 or l["pad"][1] != 0:
        raise Exception("Unhandled padding!")

    if "activation" in l and l["activation"] == "elu":
        activate = "true"
    else:
        activate = "false"

    if "additive_conv" in l and l["additive_conv"] == True:
        add_or_overwrite = "true"
    else:
        add_or_overwrite = "false"

    out_padd  = 0
    out_padhw = 0
    if "output_pad" in l:
        out_padd  = l["output_pad"][0]
        out_padhw = l["output_pad"][1]

    cores = l.get("cores", 2)
    ht    = l.get("ht",    2)

    lines = []

    #allocate layer
    params = (l["bn"], l["ifm"], l["ofm"], l["id"], l["ihw"],
              l["kernel_dim"][2], l["kernel_dim"][3],
              0, out_padd, 0, 0, out_padhw,
              activate, add_or_overwrite, cores, ht)

    params_template  = '"'
    params_template += 'BN={} IFM={} OFM={} ID={} IHW={} KD={} KHW={} '
    params_template += 'OUT_D_SKIP={} OUT_PADD={} '
    params_template += 'OUT_H_SKIP={} OUT_W_SKIP={} OUT_PADHW={} '
    params_template += 'OUT_STRIDE_D=1 OUT_STRIDE_HW=1 '
    params_template += 'ACTIVATION={} ADDOROVERWRITE={} CORES={} HT={}'
    params_template += '"'

    params_str = params_template.format(*params)

    lines.append('layers["{}"] = znn::phi::jitMakeLayer("{}", {}, this->lib_path);'.format(l["name"], l["type"], params_str))
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

    if "additive_conv" in l and l["additive_conv"]:
        lines.append('tensors["{}"] = new znn::phi::hbw_array<float>({});'.format(
                                                  l["scale"], l["scale_size"]))
        lines += fill_tensor('{}'.format(l["scale"]), l["scale_data"])

    return lines

def conv_forward_params(lparam):
    l = lparam
    params = ''
    params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
    params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["kernel"], l["bias"])

    if "additive_conv" in l and l["additive_conv"]:
        params += 'tensors["{}"]->data()'.format(l["scale"])
    else:
        params += 'NULL '

    return params
