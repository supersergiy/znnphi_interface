
def forward_layer_lines(lparams):
   lt = lparams["type"]
   l  = lparams

   lines = []
   if lt == "conv":
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["kernel"], l["bias"])
       lines.append('layers["{}"]->forward({});'.format(l["name"], params))
   elif lt in ["pool", "block_input", "unblock_output"]:
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'NULL, NULL'
   elif lt in ["bnorm"]:
       params  = 'tensors["{}"]->data(), tensors["{}"]->data(), '.format(l["bot"], l["top"])
       params += 'tensors["{}"]->data(), tensors["{}"]->data()'.format(l["scale"], l["bias"])
       lines.append('layers["{}"]->forward({});'.format(l["name"], params))

   return lines 
   '''
   elif lt == "ELU":
   elif lt == "deconv":
   elif lt == "Sigmoid":
   elif lt == "BatchNorm":
   elif lt == "Scale":
   elif lt == "Eltwise":
      #TODO: do I need to reset dimensions here?
   else:
      raise Exception("Unsupported Layer: {}".format(lt))'''
