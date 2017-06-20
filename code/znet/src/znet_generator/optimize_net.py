from tensor import Tensor

#TODO: fix this shitty code
def generate_layer_order_info(net):
    tensors, layer_info, layer_order  = net
    last_toucher = {}
    for lname in layer_order:
        l = layer_info[lname]
        bot = l["bot"]

        if isinstance(bot, list):
            for b in bot:
                prev_name = last_toucher[b]
                prev_l    = layer_info[prev_name] 

                if "next" in prev_l:
                    prev_l["next"] = "many"
                else:
                    prev_l["next"] = lname              
                    l["prev"]      = prev_name

            pass #TODO
        else:
            if bot in last_toucher: 
                prev_name = last_toucher[bot]
                prev_l    = layer_info[prev_name] 
                if "next" in prev_l:
                    prev_l["next"] = "many"
                else:
                    prev_l["next"] = lname              
                    l["prev"]      = prev_name

        last_toucher[l["top"]] = lname

def expand_convs(net):
    tensors, layer_info, layer_order  = net
    for lname in list(layer_order):#conversion to list is necessary to "freeze" the deque, since some elements will be removed
        if lname in layer_info:
            l  = layer_info[lname]
            lt = l["type"]

            if lt == "conv":
                next_name = l["next"]
                if next_name == "many":
                    continue
                next_l    = layer_info[next_name]
                while next_l["type"] in ["scale", "bnorm", "elu"]:
                    if next_l["type"] in ["scale", "bnorm"]:
                        consume_scale(layer_info, lname, next_name)
                    else:#if next_l["type"] == "elu":
                        consume_elu(layer_info, lname, next_name)
                    #update the next link
                    l["next"] = next_l["next"]
                    #remove the consumed layer
                    print "Removing {}!".format(next_name)
                    del layer_info[next_name]
                    order_of_next = layer_order.index(next_name)
                    layer_order.remove(next_name)
                    #update the local shortcuts
                    next_name = l["next"]
                    if next_name == "many":
                        break
                    next_l    = layer_info[next_name]

def consume_scale(layer_info, lname, next_name):
    next_l = layer_info[next_name]
    l      = layer_info[lname]

    kernel = l["kernel_data"]
    bias   = l["bias_data"]

    scale_multipliers = next_l["scale_data"]
    scale_bias        = next_l["bias_data"]

    for ofm in range(l["ofm"]):
        kernel[:][ofm][:][:][:] *= scale_multipliers[ofm]
        if not bias is None:
            bias[ofm] *= scale_multipliers[ofm]
            bias[ofm] += scale_bias[ofm]
    if bias is None:
        bias = scale_bias

def consume_elu(layer_info, lname, next_name):
    l = layer_info[lname]
    l["activation"] = "elu"

def handle_padding(net):
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
            
            l["id"]  =  padded_dim[2]
            l["ihw"] = padded_dim[3]
            l["pad"] = [0, 0, 0]
            #add pad layer 
            convs_order = layer_order.index(lname)
            layer_order.insert(convs_order - 1, padder_name)
            layer_info[padder_name] = pad_param


def optimize_net(net):
    generate_layer_order_info(net)
    expand_convs(net)
    handle_padding(net)

