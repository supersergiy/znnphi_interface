from .conv    import allocate_conv_lines
from .deconv  import allocate_deconv_lines
from .pool    import allocate_pool_lines
from .bnorm   import allocate_bnorm_lines
from .elu     import allocate_elu_lines
from .relu    import allocate_relu_lines
from .sigmoid import allocate_sigmoid_lines
from .pad     import allocate_pad_lines
from .scale   import allocate_scale_lines
from .slc     import allocate_slc_lines
from .eltwise import allocate_eltwise_lines
from .merge   import allocate_merge_lines
from .crop    import allocate_crop_lines

from .block_input    import allocate_block_input_lines
from .unblock_output import allocate_unblock_output_lines

def allocate_layer_lines(lparams, core_options, cpu_offset):
   lparams["cpu_offset"] = cpu_offset
   lt = lparams["type"]

   if lt in core_options:
       lparams["cores"] = core_options[lt][0]
       lparams["ht"]    = core_options[lt][1]

   if lt == "conv":
      return allocate_conv_lines(lparams)
   if lt == "deconv":
      return allocate_deconv_lines(lparams)
   elif lt == "pool":
      return allocate_pool_lines(lparams)
   elif lt == "block_input":
      return allocate_block_input_lines(lparams)
   elif lt == "unblock_output":
      return allocate_unblock_output_lines(lparams)
   elif lt == "bnorm":
      return allocate_bnorm_lines(lparams)
   elif lt == "elu":
      return allocate_elu_lines(lparams)
   elif lt == "relu":
      return allocate_relu_lines(lparams)
   elif lt == "scale":
      return allocate_scale_lines(lparams)
   elif lt == "slc":
      return allocate_slc_lines(lparams)
   elif lt == "eltwise":
      return allocate_eltwise_lines(lparams)
   elif lt == "pad":
      return allocate_pad_lines(lparams)
   elif lt == "crop":
      return allocate_crop_lines(lparams)
   elif lt == "merge":
      return allocate_merge_lines(lparams)
   elif lt == "sigmoid":
      return allocate_sigmoid_lines(lparams)
   elif lt == "input":
      return []
   elif lt == "dummy_data":
       return []
   else:
      raise Exception("Unsupported Layer: {}".format(lt))
   '''
   elif lt == "Sigmoid":
   else:
       '''
