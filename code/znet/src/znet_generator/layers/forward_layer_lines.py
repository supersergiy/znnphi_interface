
def forward_layer_lines(lparams):
   lt = lparams["type"]
   l  = lparams

   lines = []
   if lt in ["conv"]:
       lines += forward_conv_lies()
   elif lt in ["deconv"]:
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["kernel"], l["bias"])
       lines.append('layers["{}"]->forward({});'.format(l["name"], params))
   elif lt in ["pool", "block_input", "unblock_output", "elu"]:
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'NULL, NULL'
       lines.append('layers["{}"]->forward({});'.format(l["name"], params))
   elif lt in ["bnorm", "scale"]:
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["scale"], l["bias"])
       lines.append('layers["{}"]->forward({});'.format(l["name"], params))
   elif lt in ["eltwise"]:
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"][0], l["top"])
       params += 'tensors["{}"]->data(), NULL'.format(l["bot"][1]) 
       lines.append('layers["{}"]->forward({});'.format(l["name"], params))

   return lines 
  
   '''elif lt == "deconv":
   elif lt == "Sigmoid":
   elif lt == "Eltwise":
      #TODO: do I need to reset dimensions here?
   else:
      raise Exception("Unsupported Layer: {}".format(lt))'''
