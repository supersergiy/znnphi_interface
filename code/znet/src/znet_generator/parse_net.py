import json
from operator import mul

from tensor import Tensor
from layers import parse_layer, set_layer_dim

def upd_tensor(tensors, name, dim):
    size = reduce(mul, dim)
    if name not in tensors:
        tensors[name] = Tensor(dim)
    elif size > tensors[name].size:
        tensors[name].size = size

def parse_net(net_path):
   with open(net_path) as f:
       net = json.load(f)

   json_layers  = net["layer"]
   tensors      = {}
   layer_order  = [] 
   layer_info   = {}
   net = (tensors, layer_info, layer_order)

   for l in json_layers:
      name = l["name"]
      layer_info[name] = parse_layer(l)
      layer_order.append(name)

   for name in layer_order:
      lparams = layer_info[name]
      bot_tensor = None
      if lparams["bot"]:
         if isinstance(lparams["bot"], list):
             bot_tensor = tensors[lparams["bot"][0]] #TODO: possible source of unlimited future bugs
         else:
             bot_tensor = tensors[lparams["bot"]]

      set_layer_dim(lparams, bot_tensor)

      top_name = lparams["top"]
      upd_tensor(tensors, top_name, lparams["top_dim"])

   add_block_input(net)
   add_unblock_output(net)

   return net

def add_block_input(net):
    tensors, layer_info, layer_order = net

    block_params = {
                        "type": "block_input",
                        "name": "block_input",
                        "bot": "user_input",
                        "top": "input", 
                        "bot_dim": tensors["input"].dim
                   }
    layer_order.insert(0, "block_input")
    layer_info["block_input"] = block_params
    tensors["user_input"] = Tensor(block_params["bot_dim"])

def add_unblock_output(net):
    tensors, layer_info, layer_order = net
    if "output" in tensors:
        unblock_params = {}
        unblock_params = {
                            "type": "unblock_output",
                            "name": "unblock_output",
                            "bot": "output",
                            "top": "user_output",
                            "bot_dim": tensors["output"].dim
                       }
        layer_order.append("unblock_output")
        layer_info["unblock_output"] = unblock_params
        tensors["user_output"] = Tensor(unblock_params["bot_dim"])
    else:
        print "WARNING: there's no output tensor in your network"
        exit()


