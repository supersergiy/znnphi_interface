from .conv import conv_forward_params

def forward_layer_lines(lparams):
   lt = lparams["type"]
   l  = lparams

   lines = []
   params = ''
   if lt in ["conv", "deconv"]:
       params = conv_forward_params(l)
   elif lt in ["pool", "block_input", "unblock_output", "elu", "relu", "pad", "sigmoid", "slc"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'NULL, NULL'
   elif lt in ["crop"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"][0], l["top"])
       params += 'NULL, NULL'
   elif lt in ["bnorm", "scale"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["scale"], l["bias"])
   elif lt in ["eltwise", "merge"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"][0], l["top"])
       params += 'tensors["{}"]->data(), NULL'.format(l["bot"][1])
   elif lt in ["neweltwise"]:
       num_bots_name = 'num_bots_{}'.format(l["name"])
       num_bots = 'int {} = {};'.format(num_bots_name, len(l["bot"]))
       bots_pointers = ', '.join(['tensors["{}"]->data()'.format(t_name) for t_name in l["bot"]])
       bots_array_name = 'bots_{}'.format(l["name"])
       bots_array = 'void* {}[{}] = {{ {} }};'.format(bots_array_name, num_bots_name, bots_pointers)
       lines.append(num_bots)
       lines.append(bots_array)
       params += '{}, &{}, tensors["{}"]->data(), '.format(bots_array_name, num_bots_name, l["top"])
       params += 'NULL'
   elif lt in ["input", "dummy_data"]:
       return []

   lines.append('layers["{}"]->forward({});'.format(l["name"], params))
   return lines
