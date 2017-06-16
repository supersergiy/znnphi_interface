from conv   import allocate_conv_lines 
from pool   import allocate_pool_lines 
from bnorm  import allocate_bnorm_lines
from block_input    import allocate_block_input_lines
from unblock_output import allocate_unblock_output_lines

def allocate_layer_lines(lparams):
   lt = lparams["type"]
   if lt == "conv":
      return allocate_conv_lines(lparams) 
   elif lt == "pool":
      return allocate_pool_lines(lparams) 
   elif lt == "block_input":
      return allocate_block_input_lines(lparams) 
   elif lt == "unblock_output":
      return allocate_unblock_output_lines(lparams) 
   elif lt == "bnorm":
      return allocate_bnorm_lines(lparams) 
   else:
      return [] 
   '''
   elif lt == "ELU":
   elif lt == "deconv":
   elif lt == "Sigmoid":
   elif lt == "Scale":
   elif lt == "Eltwise":
      #TODO: do I need to reset dimensions here?
   else:
      raise Exception("Unsupported Layer: {}".format(lt))'''
