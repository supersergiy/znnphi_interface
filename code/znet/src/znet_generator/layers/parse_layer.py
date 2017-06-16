from conv   import parse_conv
from deconv import parse_deconv
from pool   import parse_pool
from bnorm  import parse_bnorm 

def check_params(lparams):
   necessary_fields = ["top", "bot", "name", "type"]
   for f in necessary_fields:
      if not f in lparams.keys():    
         import pdb; pdb.set_trace()
         raise Exception("Bad parameter parsing")

def parse_layer(l):
   lparams = {}
   lt = l["type"]

   if lt == "Input":
      dim = l["input_param"]["shape"][0]["dim"]
      lparams["type"] = "input"
      lparams["name"] = "input"
      lparams["top_dim"] = dim
      lparams["top"] = "input"
      lparams["bot"] = None # is actually fetched from user_input
                            # by a blocker layer
   elif lt == "Convolution":
      lparams = parse_conv(l)
   elif lt == "Deconvolution":
      lparams = parse_deconv(l)
   elif lt == "Pooling":
      lparams = parse_pool(l)
   elif lt == "BatchNorm":
      lparams = parse_bnorm(l)
   elif lt == "ELU":
      lparams["type"] = "elu"
      lparams["top"] = l["top"][0]
      lparams["bot"] = l["bottom"][0]
      lparams["name"] = l["name"]
   elif lt == "Sigmoid":
      lparams["type"] = "sigmoid"
      lparams["top"] = l["top"][0]
      lparams["bot"] = l["bottom"][0]
      lparams["name"] = l["name"]
   elif lt == "Scale":
      lparams["type"] = "scale"
      lparams["top"] = l["top"][0]
      lparams["bot"] = l["bottom"][0]
      lparams["name"] = l["name"]
   elif lt == "Eltwise":
      lparams["type"] = "eltwise"
      lparams["top"] = l["top"][0]
      lparams["bot"] = l["bottom"][0]
      lparams["name"] = l["name"]
      #TODO: do I need to reset dimensions here?
   else:
      raise Exception("Unsupported Layer: {}".format(lt))

   check_params(lparams)
   return lparams
