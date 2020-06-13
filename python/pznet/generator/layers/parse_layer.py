from .conv    import parse_conv
from .deconv  import parse_deconv
from .deconv  import parse_deconv
from .pool    import parse_pool
from .bnorm   import parse_bnorm
from .elu     import parse_elu
from .relu    import parse_relu
from .scale   import parse_scale
from .bias    import parse_bias
from .eltwise import parse_eltwise
from .slc     import parse_slc
from .crop    import parse_crop
from .merge   import parse_merge
from .mergecrop   import parse_mergecrop

def check_params(lparams):
   necessary_fields = ["top", "bot", "name", "type"]
   for f in necessary_fields:
      if not f in lparams.keys():
         import pdb; pdb.set_trace()
         raise Exception("Bad parameter parsing")

def parse_layer(l, arch):
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
   elif lt == "DummyData":
      dim = l["dummy_data_param"]["shape"][0]["dim"]
      lparams["type"] = "dummy_data"
      lparams["name"] = l["name"]
      lparams["top_dim"] = dim
      lparams["top"] = l["top"][0]
      lparams["bot"] = None # is actually fetched from user_input
                            # by a blocker layer
   elif lt == "Crop":
      lparams = parse_crop(l, arch)
   elif lt == "Merge":
      lparams = parse_merge(l, arch)
   elif lt == "MergeCrop":
      lparams = parse_mergecrop(l, arch)
   elif lt == "Convolution":
      lparams = parse_conv(l, arch)
   elif lt == "Deconvolution":
      lparams = parse_deconv(l, arch)
   elif lt == "Pooling":
      lparams = parse_pool(l, arch)
   elif lt == "BatchNorm":
      lparams = parse_bnorm(l, arch)
   elif lt == "ELU":
      lparams = parse_elu(l, arch)
   elif lt == "ReLU":
      lparams = parse_relu(l, arch)
   elif lt == "Scale":
      lparams = parse_scale(l, arch)
   elif lt == "Bias":
      lparams = parse_bias(l, arch)
   elif lt == "Slice":
      lparams = parse_slc(l, arch)
   elif lt == "Eltwise":
      lparams = parse_eltwise(l, arch)
   elif lt == "Sigmoid":
      lparams["type"] = "sigmoid"
      lparams["top"] = l["top"][0]
      lparams["bot"] = l["bottom"][0]
      lparams["name"] = l["name"]
   else:
      raise Exception("Unsupported Layer: {}".format(lt))

   check_params(lparams)
   return lparams
