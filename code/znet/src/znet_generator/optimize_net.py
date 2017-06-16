
def generate_layer_order_info(net):
    tensors, layer_info, layer_order  = net
    last_toucher = {}
    for lname in layer_order:
        l = layer_info[lname]
        bot = l["bot"]
        if isinstance(bot, list):
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
    for lname in layer_order:
        l  = layer_info[lname]
        lt = l["type"]

        if lt == "conv":
            next_name = l["next"]
            next_l    = layer_info[next_name]
            while next_l["type"] in ["scale", "bnorm", "elu"]:
                if next_l["type"] in ["scale", "bnorm"]:
                    consume_scale(layer_info, lname, next_name)
                else:#if next_l["type"] == "elu":
                    consume_elu(layer_info, lname, next_name)
                #update the next link
                l["next"] = next_l["next"]
                #remove the consumed layer
                del layer_info[next_name]
                order_of_next = layer_order.index(next_name)
                layer_order.remove(order_of_next)
                #update the local shortcuts
                next_name = l["next"]
                next_l    = layer_info[next_name]

def consume_scale(layer_info, lname, next_name):
    next_l = layer_info[next_name]
    l      = layer_info[lname]

    kernel = l["kernel_data"]
    bias   = l["bias_data"]

    scale_multipliers = next_l["scale_data"]
    scale_bias        = next_l["bias_data"]

    for ofm in l["ofm"]:
        kernel[:][ofm][:][:][:] *= scale_multipliers[ofm]
        bias[ofm] *= scale_multipliers[ofm]
        bias[ofm] += scale_bias[ofm]

def consume_elu(layer_info, lname, next_name):
    l = layer_info[lname]
    l["activation"] = "elu"

def optimize_net(net):
    generate_layer_order_info(net)
    expand_convs(net)

