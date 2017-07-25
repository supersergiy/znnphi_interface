from tensor import Tensor
import numpy as np
import copy
from   layers import set_layer_dim

#TODO: fix this shitty code
def generate_layer_order_info(net):
    tensors, layer_info, layer_order  = net
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
    tensors, layer_info, layer_order  = net
    
    from_index = 0
    if not starting_layer is None:
        from_index = layer_order.index(starting_layer) + 1

    for i in range(from_index, len(layer_order)):
        lname = layer_order[i]
        l     = layer_info[lname]

        if isinstance(l["bot"], list):
            for i in range(len(l["bot"])):
                if l["bot"][i] == replace_from:
                    l["bot"][i] =replace_with 

        else:
            if l["bot"] == replace_from:
                l["bot"] = replace_with 
            
         
def expand_convs(net):
    tensors, layer_info, layer_order  = net
    for lname in layer_order:
        if lname in layer_info:
            l  = layer_info[lname]
            lt = l["type"]

            if lt in ["conv"] and len(l["next"]) == 1:
                next_name = l["next"][0] 
                next_l    = layer_info[next_name]

                while next_l["type"] in ["scale", "bnorm", "elu"]:
                    if next_l["type"] in ["scale", "bnorm"]:
                        consume_scale(layer_info, lname, next_name)
                    else:#if next_l["type"] == "elu":
                        consume_elu(layer_info, lname, next_name)

                    substitute_tensor(net, next_l["top"], l["top"], lname)
                    #remove the consumed layer
                    delete_layer(net, next_name, lname)
                    
                    if len(l["next"]) != 1:
                        break

                    next_name = l["next"][0]
                    next_l    = layer_info[next_name]

def delete_layer(net, layer_name, prev_layer):
    tensors, layer_info, layer_order  = net
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
        kernel[:][ofm][:][:][:] *= scale_multipliers[ofm]
        bias[ofm] *= scale_multipliers[ofm]
        bias[ofm] += scale_bias[ofm]

    
    if bias is None:
        bias = scale_bias

    if "additive_conv" in l and l["additive_conv"]:
        for ofm in range(l["ofm"]):
            l["scale_data"][ofm] *= scale_multipliers[ofm]

def consume_elu(layer_info, lname, next_name):
    l = layer_info[lname]
    l["activation"] = "elu"

def handle_padding(net):
    tensors, layer_info, layer_order  = net

    handle_implicit_paddings(net)
    insert_explicit_paddings(net) 

def remove_padding_from_conv(net, conv_name):
    tensors, layer_info, layer_order  = net
    l = layer_info[conv_name]

    l["id"]  += 2 * l["pad"][0] 
    l["ihw"] += 2 * l["pad"][1]

    l["pad"] = [0, 0, 0]


def handle_implicit_paddings(net):
    tensors, layer_info, layer_order  = net
    count = 0 

    for lname in list(layer_order):
        l = layer_info[lname]

        if l["type"] == "conv" and len(l["next"]) == 1 and len(l["prev"]) == 1: 
            next_name = l["next"][0]

            if layer_info[next_name]["type"] == "conv":
                next_l = layer_info[next_name]
                if next_l["pad"][0] != 0 or next_l["pad"][1] != 0:
                    count += 1
                    print "Output paddiing for {}!".format(l["name"])
                    l["output_pad"] = copy.copy(next_l["pad"])
                    l["top_dim"][2] += 2 * l["output_pad"][0]
                    l["top_dim"][3] += 2 * l["output_pad"][1]
                    l["top_dim"][4] += 2 * l["output_pad"][1]
                    tensors[l["top"]] = Tensor(l["top_dim"]) 
                    
                    remove_padding_from_conv(net, next_name)
                    set_layer_dim(next_l, tensors[l["top"]]) 

def insert_explicit_paddings(net):
    tensors, layer_info, layer_order  = net

    for lname in list(layer_order):
        l = layer_info[lname]
        if l["type"] == "conv" and (l["pad"][0] != 0 or l["pad"][1] != 0):
            #set up pad params
            padder_name = "{}_padder".format(lname)
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

            #rewire tensors
            pad_param["bot"] = l["bot"]
            pad_param["top"] = "{}_padded".format(l["bot"])
            tensors[pad_param["top"]] = Tensor(pad_param["top_dim"]) 

            l["bot"] = pad_param["top"]
            pad_param["top_dim"] = l["bot_dim"]
           
            #modify the conv layer params to remove padding
            remove_padding_from_conv(net, lname)

            #add pad layer 
            convs_order = layer_order.index(lname)
            layer_order.insert(convs_order, padder_name)
            layer_info[padder_name] = pad_param
            

def stride1_deconv_to_conv(net):
    tensors, layer_info, layer_order  = net
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

def eliminate_adds(net):
    tensors, layer_info, layer_order  = net
    count = 0

    for lname in (layer_order):
        l = layer_info[lname]
        if l["type"] in ["conv"]:
            if len(l["next"]) == 1 and layer_info[l["next"][0]]["type"] == "eltwise": #TODO: all eltwise are sums now, so this should be changed later
                next_name = l["next"][0]
                next_l = layer_info[next_name] 
                
                # make sure that this conv executes before the things that's added to it
                # otherwise we can't consume this add with this conv
                can_consume = True
                my_order = layer_order.index(lname)
                for added_layer in next_l["prev"]:
                    if layer_order.index(added_layer) > my_order:
                        can_consume = False

                if not can_consume:
                    continue
                        
                if l["additive_conv"] == True:
                    raise Exception("Double additive layer")

                l["additive_conv"]  = True
                l["scale_data"]     = [1] * l["ofm"]

                #set the top of the current layer to the thing to add to
                if next_l["bot"][0] == l["top"]:
                    l["top"] = next_l["bot"][1]
                else:
                    l["top"] = next_l["bot"][0]
                 

                for sum_receiver in next_l["next"]:
                    #TODO: check if it's another sum, then handle differently
                    layer_info[sum_receiver]["bot"] = l["top"]

                #remove eltwise
                delete_layer(net, next_name, lname)

def optimize_net(net):
    generate_layer_order_info(net)
    stride1_deconv_to_conv(net)
    eliminate_adds(net)
    expand_convs(net)
    handle_padding(net)


