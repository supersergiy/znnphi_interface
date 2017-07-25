from conv import conv_forward_params

def forward_layer_lines(lparams):
   lt = lparams["type"]
   l  = lparams

   lines = []
   params = '' 
   if lt in ["conv", "deconv"]:
       params = conv_forward_params(l)
   if lt in ["aadeconv"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["kernel"], l["bias"])
   elif lt in ["pool", "block_input", "unblock_output", "elu", "pad", "sigmoid"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'NULL, NULL'
   elif lt in ["bnorm", "scale"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["scale"], l["bias"])
   elif lt in ["eltwise"]:
       params += 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"][0], l["top"])
       params += 'tensors["{}"]->data(), NULL'.format(l["bot"][1]) 
   elif lt in ["input"]:
       return lines

   lines.append('layers["{}"]->forward({});'.format(l["name"], params))
   return lines 
  
   '''elif lt == "deconv":
   elif lt == "Sigmoid":
   elif lt == "Eltwise":
      #TODO: do I need to reset dimensions here?
   else:
      raise Exception("Unsupported Layer: {}".format(lt))'''
