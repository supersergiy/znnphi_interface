from tensor import Tensor
import numpy as np
import copy
from   layers import set_layer_dim, round_to_simd


#TODO: fix this shitty code
def generate_layer_order_info(net):
    tensors, layer_info, layer_order, misc = net
    last_toucher = {}

    for lname in layer_order:
        l = layer_info[lname]

        if "next" not in l or "prev" not in l:
            l["next"] = []
            l["prev"] = []
        bot = l["bot"]

        if not l["bot"] in [None, "user_input"]:
            if isinstance(bot, list):
                for b in bot:
                    prev_name = last_toucher[b]
                    prev_l    = layer_info[prev_name]

                    prev_l["next"].append(lname)
                    l["prev"].append(prev_name)
            else:
                prev_name = last_toucher[bot]
                prev_l    = layer_info[prev_name]

                prev_l["next"].append(lname)
                l["prev"].append(prev_name)

        last_toucher[l["top"]] = lname

def substitute_tensor(net, replace_from, replace_with, starting_layer=None):
    tensors, layer_info, layer_order, misc = net

    from_index = 0
    if not starting_layer is None:
        from_index = layer_order.index(starting_layer) + 1

    for i in range(from_index, len(layer_order)):
        lname = layer_order[i]
        l     = layer_info[lname]

        if isinstance(l["bot"], list):
            for i in range(len(l["bot"])):
                if l["bot"][i] == replace_from:
                    l["bot"][i] = replace_with

        else:
            if l["bot"] == replace_from:
                l["bot"] = replace_with
        if l["top"] == replace_from:
            l["top"] = replace_with

def generate_dummy_scale_params(prev_lparam):
    l = prev_lparam
    name = "{}_dummy_scale".format(prev_lparam["name"])

    param   = {}
    param["name"]  = name
    param["type"]  = "scale"
    param["bn"]    = l["bn"]
    param["ifm"]   = l["ifm"]
    param["id"]    = l["id"]
    param["ihw"]   = l["ihw"]

    param["top_dim"] = l["top_dim"]
    param["bot_dim"] = l["top_dim"]

    param["bot"] = l["bot"]
    param["top"] = "{}_dummy".format(l["bot"])

    param["scale"] = "scale_{}".format(name)
    param["bias"]  = "bias_{}".format(name)

    param["scale_data"] = [1.0] * param["ifm"]
    param["bias_data"]  = [0.0] * param["ifm"]
    param["scale_size"] = round_to_simd(param["ifm"], l["arch"])
    param["bias_size"] = round_to_simd(param["ifm"], l["arch"])
    return param

def expand_convs(net, opt_param):
    tensors, layer_info, layer_order, misc = net
    count = 0

    target_expansions = []
    if 'lin_fuse' in opt_param:
        target_expansions += ['scale', 'bnorm']
    if 'act_fuse' in opt_param:
        target_expansions += ['elu', 'relu']
    for lname in layer_order:
        if lname in layer_info:
            l  = layer_info[lname]
            lt = l["type"]

            if lt in ["conv","deconv"] and len(l["next"]) == 1:
                next_name = l["next"][0]
                next_l    = layer_info[next_name]
                while next_l["type"] in target_expansions:
                    if next_l["type"] == "bnorm" and next_l["static_bnorm"] == False:
                        break
                    if next_l["type"] in ["scale", "bnorm"]:
                        consume_scale(layer_info, lname, next_name)
                    else:#if next_l["type"] == "elu":
                        consume_activation(layer_info, lname, next_name)

                    substitute_tensor(net, next_l["top"], l["top"], lname)
                    #remove the consumed layer
                    delete_layer(net, next_name, lname)

                    if len(l["next"]) != 1:
                        break

                    next_name = l["next"][0]
                    next_l    = layer_info[next_name]


def insert_layer(net, lparam, prev_lname):
    lname = lparam["name"]
    tensors, layer_info, layer_order, misc = net

    prev_order = layer_order.index(prev_lname)
    layer_order.insert(prev_order, lname)
    layer_info[lname] = lparam


def delete_layer(net, layer_name, prev_layer):
    tensors, layer_info, layer_order, misc = net
    l = layer_info[layer_name]

    print "Removing {}!".format(layer_name)
    for prev_name in l["prev"]:
        layer_info[prev_name]["next"].remove(layer_name)

        if prev_name == prev_layer:
            layer_info[prev_name]["next"] += l["next"]
        else:
            layer_info[prev_name]["next"].append(prev_layer)


    for next_name in l["next"]:
        layer_info[next_name]["prev"].remove(layer_name)
        #TODO: make sure it's not another add
        layer_info[next_name]["prev"].append(prev_layer)

    if l["type"] == "eltwise":
        layer_info[prev_layer]["prev"] += [p for p in l["prev"] if p != prev_layer]
    del layer_info[layer_name]

    layer_order.remove(layer_name)

def consume_scale(layer_info, lname, next_name):
    next_l = layer_info[next_name]
    l      = layer_info[lname]


    if l["bias_data"] is None:
        l["bias_data"] = np.zeros(l["ofm"], dtype=np.float)

    kernel = l["kernel_data"]
    bias   = l["bias_data"]

    scale_multipliers = next_l["scale_data"]
    scale_bias        = next_l["bias_data"]
    for ofm in range(l["ofm"]):
        if l["type"] == "conv":
            kernel[ofm,:,:,:,:] *= scale_multipliers[ofm]
            bias[ofm] *= scale_multipliers[ofm]
            bias[ofm] += scale_bias[ofm]
        elif l["type"] == "deconv":
            kernel[:,ofm,:,:,:] *= scale_multipliers[ofm]
            bias[ofm] *= scale_multipliers[ofm]
            bias[ofm] += scale_bias[ofm]


    if "additive_conv" in l and l["additive_conv"]:
        for ofm in range(l["ofm"]):
            if l["type"] in ["conv", "deconv"]:
                l["scale_data"][ofm] *= scale_multipliers[ofm]

def consume_activation(layer_info, lname, next_name):
    l = layer_info[lname]
    l["activation"] = layer_info[next_name]["type"]

def handle_padding(net, opt_param):
    handle_conv_padding(net, opt_param)
    handle_deconv_padding(net)

def handle_deconv_padding(net):
    tensors, layer_info, layer_order, misc = net
    for lname in list(layer_order):
        l  = layer_info[lname]
        lt = l["type"]
        if lt == "deconv" and l["pad"] != [0, 0, 0]:
            #change the output dims of deconv to have the full output range
            #insert a crop right after
            crop_param = generate_crop_param(l)
            #create the in between tensor of size: precrop
            tensors[crop_param["bot"][0]] = Tensor(crop_param["bot_dim"], l["arch"])

            #rewire deconv to send input to cropper
            l["top"] = crop_param["bot"][0]

            remove_padding_from_deconv(net, lname)

            #add crop layer
            insert_layer(net, crop_param, prev_lname=l["next"][0])


def handle_conv_padding(net, opt_param):
    if 'implicit_pad' in opt_param:
        insert_implicit_conv_pads(net)
    insert_explicit_conv_pads(net)


def remove_padding_from_conv(net, conv_name):
    tensors, layer_info, layer_order, misc = net
    l = layer_info[conv_name]

    l["id"]  += 2 * l["pad"][0]
    l["ihw"] += 2 * l["pad"][1]
    l["pad"] = [0, 0, 0]

def remove_padding_from_deconv(net, deconv_name):
    tensors, layer_info, layer_order, misc = net
    l = layer_info[deconv_name]

    l["top_dim"][2] += 2 * l["pad"][0]
    l["top_dim"][3] += 2 * l["pad"][1]
    l["top_dim"][4] += 2 * l["pad"][1]

    l["pad"] = [0, 0, 0]


def insert_implicit_conv_pads(net):
    tensors, layer_info, layer_order, misc = net
    count = 0

    for lname in list(layer_order):
        l = layer_info[lname]

        if l["type"] == "conv" and len(l["next"]) == 1 and len(l["prev"]) == 1:
            next_name = l["next"][0]

            if layer_info[next_name]["type"] == "conv":
                next_l = layer_info[next_name]
                if next_l["pad"][0] != 0 or next_l["pad"][1] != 0:
                    count += 1
                    print "Output paddinig for {}!".format(l["name"])
                    l["output_pad"] = copy.copy(next_l["pad"])
                    l["top_dim"][2] += 2 * l["output_pad"][0]
                    l["top_dim"][3] += 2 * l["output_pad"][1]
                    l["top_dim"][4] += 2 * l["output_pad"][1]
                    tensors[l["top"]] = Tensor(l["top_dim"], l["arch"])

                    remove_padding_from_conv(net, next_name)
                    set_layer_dim(next_l, tensors[l["top"]])

def generate_crop_param(deconv_lparam):
    assert deconv_lparam["type"] == "deconv"
    l = deconv_lparam

    crop_param   = {}
    crop_name = "{}_crop".format(l["name"])
    crop_param["name"]  = crop_name
    crop_param["type"]  = "crop"

    crop_param["bn"]    = l["bn"]
    crop_param["ofm"]   = l["ofm"]
    crop_param["ifm"]   = l["ofm"]

    crop_param["id"]    = l["stride"][0] * (l["id"]  - 1) + l["kernel_dim"][2]
    crop_param["ihw"]   = l["stride"][1] * (l["ihw"] - 1) + l["kernel_dim"][3]
    crop_param["od"]    = l["stride"][0] * (l["id"]  - 1) + l["kernel_dim"][2] - 2*l["pad"][0]
    crop_param["ohw"]   = l["stride"][1] * (l["ihw"] - 1) + l["kernel_dim"][3] - 2*l["pad"][1]

    crop_param["top_dim"] = copy.deepcopy(l["top_dim"])
    crop_param["bot_dim"] = [ crop_param["bn"], crop_param["ifm"], crop_param["id"], crop_param["ihw"], crop_param["ihw"] ]

    crop_param["z_offset"]  = l["pad"][0]
    crop_param["xy_offset"] = l["pad"][1]

    crop_param["bot"] = ["{}_precrop".format(l["top"])]
    crop_param["top"] = l["top"]

    return crop_param

def generate_pad_param(conv_lparam):
    assert conv_lparam["type"] == "conv"
    l = conv_lparam
    padder_name = "{}_padder".format(conv_lparam["name"])

    pad_param   = {}
    pad_param["name"]  = padder_name
    pad_param["type"]  = "pad"
    pad_param["bn"]    = l["bn"]
    pad_param["ifm"]   = l["ifm"]
    pad_param["id"]    = l["id"]
    pad_param["ihw"]   = l["ihw"]
    pad_param["padd"]  = l["pad"][0]
    pad_param["padhw"] = l["pad"][1]

    padded_dim = [ l["bn"], l["ifm"],
                   l["id"]+2*l["pad"][0],
                   l["ihw"]+2*l["pad"][1],
                   l["ihw"]+2*l["pad"][1] ]

    pad_param["top_dim"] = padded_dim

    pad_param["bot"] = l["bot"]
    pad_param["top"] = "{}_padded".format(l["name"])
    return pad_param

def insert_explicit_conv_pads(net):
    tensors, layer_info, layer_order, misc = net
    for lname in list(layer_order):
        l = layer_info[lname]
        if l["type"] == "conv" and (l["pad"][0] != 0 or l["pad"][1] != 0):
            pad_param = generate_pad_param(l)
            #alocate the padder out tensor
            tensors[pad_param["top"]] = Tensor(pad_param["top_dim"], l["arch"])

            #rewire conv to take input from padder
            l["bot"] = pad_param["top"]
            pad_param["top_dim"] = l["bot_dim"]

            #modify the conv layer params to remove padding
            remove_padding_from_conv(net, lname)

            #add pad layer
            insert_layer(net, pad_param, prev_lname=lname)

def stride1_deconv_to_conv(net):
    tensors, layer_info, layer_order, misc = net
    for lname in list(layer_order):
        l = layer_info[lname]
        if l["type"] == "deconv" and l["stride"] == [1,1,1]:
            l["type"] = "conv"

            #add padding to the input for equivalance
            for i in range(3):
                l["pad"][i] = (l["json_kernel_size"][i] - 1) - l["pad"][i]
                if l["pad"][i] < 0:
                    raise Exception("Deconv negative padding is not supported yet")

            #conv kernel is OFMxIFMxZXY, deconv kernel is IFMxOFMxZXY
            l["kernel_dim"][0] = l["ofm"]
            l["kernel_dim"][1] = l["ifm"]

            l["kernel_data"] = conv_to_deconv_kernel(l["kernel_data"])

def conv_to_deconv_kernel(conv_kernel):
    deconv_kernel = np.swapaxes(conv_kernel, 0, 1)
    deconv_kernel = np.flip(deconv_kernel, 2)
    deconv_kernel = np.flip(deconv_kernel, 3)
    deconv_kernel = np.flip(deconv_kernel, 4)
    return deconv_kernel

def expand_mergecrops(net):
    tensors, layer_info, layer_order, misc = net
    count = 0

    for lname in (layer_order):
        l = layer_info[lname]
        if l["type"] in ["mergecrop"]:
            #make a crop
            crop_param = copy.deepcopy(l)
            crop_param["type"] = "crop"
            crop_param["name"] = "{}_crop".format(l["name"])
            crop_param["bot"] = [l["bot"][1]]
            crop_param["top"] = crop_param["name"]
            crop_param["top_dim"] = l["crop_top_dim"]
            crop_param["ihw"] = l["ihw2"]
            crop_param["id"]  = l["id2"]
            crop_param["ifm"]  = l["ifm2"]
            crop_param["ofm"]  = l["ifm2"]
            crop_param["ohw"] = l["ihw1"]
            crop_param["od"]  = l["id1"]
            tensors[crop_param["top"]] = Tensor(crop_param["top_dim"], l["arch"])

            #make me merge
            l["type"] = "merge"
            l["bot"][1] = crop_param["top"]
            #add the crop layer
            insert_layer(net, crop_param, prev_lname=l["name"])


def eliminate_adds(net):
    tensors, layer_info, layer_order, misc = net
    count = 0

    for lname in (layer_order):
        l = layer_info[lname]
        if l["type"] in ["conv", "deconv"]:
            if len(l["next"]) == 1 and layer_info[l["next"][0]]["type"] == "eltwise": #TODO: this works only for summing eltwise
                next_name = l["next"][0]
                next_l = layer_info[next_name]
                # make sure that this conv executes before the thing that's added to it
                # otherwise we can't consume this add with this conv
                can_consume = True
                my_order = layer_order.index(lname)
                for added_layer in next_l["prev"]:
                    if layer_order.index(added_layer) > my_order:
                        can_consume = False

                if not can_consume:
                    continue
                if "additive_conv" in l and l["additive_conv"] == True:
                    raise Exception("Double additive layer")

                l["additive_conv"]  = True
                l["scale_data"]     = [1] * round_to_simd(l["ofm"], l["arch"])

                #set the top of the current layer to the thing to add to
                if next_l["bot"][0] == l["top"]:
                    l["top"] = next_l["bot"][1]
                else:
                    l["top"] = next_l["bot"][0]

                #rename the input for the layers that use the sum output
                substitute_tensor(net, next_l["top"], l["top"], lname)

                #remove eltwise
                delete_layer(net, next_name, lname)

def optimize_net(net, opt_flags):
    #parse opt flags
    opt_param = []
    '''if not ',no_lin,' in opt_flags:
        opt_param += ["lin_fuse"]
    if not ',no_act,' in opt_flags:
        opt_param += ["act_fuse"]
    if not ',no_add,' in opt_flags:
        opt_param += ["add_fuse"]
    if not ',no_pad,' in opt_flags:
        opt_param += ["implicit_pad"]'''
    generate_layer_order_info(net)
    stride1_deconv_to_conv(net)
    expand_mergecrops(net)
    if 'add_fuse' in opt_param:
        eliminate_adds(net)
    expand_convs(net, opt_param)
    handle_padding(net, opt_param)

