from .conv    import set_conv_dim
from .deconv  import set_deconv_dim
from .pool    import set_pool_dim
from .bnorm   import set_bnorm_dim
from .elu     import set_elu_dim
from .sigmoid import set_sigmoid_dim
from .scale   import set_scale_dim
from .slc     import set_slc_dim
from .eltwise import set_eltwise_dim
from .crop    import set_crop_dim
from .mergecrop  import set_mergecrop_dim
from .dummy_data import set_dummy_data_dim

def set_layer_dim(lparams, bot_tensors):
   lt = lparams["type"]
   if lt == "input":
      return
   elif lt == "conv":
      set_conv_dim(lparams, bot_tensors)
   elif lt == "deconv":
      set_deconv_dim(lparams, bot_tensors)
   elif lt == "pool":
      set_pool_dim(lparams, bot_tensors)
   elif lt == "bnorm":
      set_bnorm_dim(lparams, bot_tensors)
   elif lt == "elu":
      set_elu_dim(lparams, bot_tensors)
   elif lt == "sigmoid":
      set_sigmoid_dim(lparams, bot_tensors)
   elif lt == "scale":
      set_scale_dim(lparams, bot_tensors)
   elif lt == "slc":
      set_slc_dim(lparams, bot_tensors)
   elif lt == "eltwise":
      set_eltwise_dim(lparams, bot_tensors)
   elif lt == "mergecrop":
      set_mergecrop_dim(lparams, bot_tensors)
   elif lt == "crop":
      set_crop_dim(lparams, bot_tensors)
   elif lt == "dummy_data":
      pass
   else:
      lparams["top_dim"] = bot_tensors.dim
   '''elif lt == "ELU":
   elif lt == "Sigmoid":
   elif lt == "BatchNorm":
   elif lt == "Scale":
   elif lt == "Eltwise":
      #TODO: do I need to reset dimensions here?
   else:
      raise Exception("Unsupported Layer: {}".format(lt))'''
