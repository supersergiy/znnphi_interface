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
                    order_of_next = list(layer_order).index(next_name)
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

def make_elus_inplace(net):
    tensors, layer_info, layer_order  = net

    for lname in layer_order:
        l = layer_info[lname]
        if l["type"] == "elu":
            import pdb; pdb.set_trace()
            rename_to   = l["bot"]
            rename_from = l["top"]
            print "Renaming {} to {}...".format(rename_from, rename_to)

            l["top"] = l["bot"]

            for other_lname in layer_order:
               other_l = layer_info[other_lname] 
               if other_l["bot"] == rename_from:
                   other_l["bot"] = rename_to
            del tensors[rename_from]


def optimize_net(net):
    generate_layer_order_info(net)
    expand_convs(net)

