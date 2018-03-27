import json
from operator import mul

from tensor import Tensor
from layers import parse_layer, set_layer_dim

def upd_tensor(tensors, name, dim, arch):
    size = reduce(mul, dim)
    if name not in tensors:
        tensors[name] = Tensor(dim, arch)
    elif size > tensors[name].size:
        tensors[name].size = size

def parse_net(net_path, arch):
   with open(net_path) as f:
       net = json.load(f)

   json_layers  = net["layer"]
   tensors      = {}
   layer_order  = []
   layer_info   = {}
   misc         = {}
   net = (tensors, layer_info, layer_order, misc)

   for l in json_layers:
      name = l["name"]
      layer_info[name] = parse_layer(l, arch)
      layer_order.append(name)

   for name in layer_order:
      lparams = layer_info[name]
      bot_tensors = None
      if lparams["bot"]:
         if isinstance(lparams["bot"], list):
             bot_tensors = [tensors[k] for k in lparams["bot"]]
         else:
             bot_tensors = tensors[lparams["bot"]]
      set_layer_dim(lparams, bot_tensors)

      top_name = lparams["top"]
      upd_tensor(tensors, top_name, lparams["top_dim"], arch)

   misc["output_tensor_name"] = layer_info[layer_order[-1]]["top"]

   add_block_input(net, arch)
   add_unblock_output(net, arch)
   return net

def add_block_input(net, arch):
    tensors, layer_info, layer_order, misc = net
    block_params = {
                        "type": "block_input",
                        "name": "block_input",
                        "bot": "user_input",
                        "top": "input",
                        "bot_dim": tensors["input"].dim
                   }
    layer_order.insert(0, "block_input")
    layer_info["block_input"] = block_params
    tensors["user_input"] = Tensor(block_params["bot_dim"], arch)

def add_unblock_output(net, arch):
    tensors, layer_info, layer_order, misc = net
    unblock_params = {}
    unblock_params = {
                        "type": "unblock_output",
                        "name": "unblock_output",
                        "bot": misc["output_tensor_name"],
                        "top": "user_output",
                        "bot_dim": tensors[misc["output_tensor_name"]].dim
                   }
    layer_order.append("unblock_output")
    layer_info["unblock_output"] = unblock_params
    tensors["user_output"] = Tensor(unblock_params["bot_dim"], arch)


