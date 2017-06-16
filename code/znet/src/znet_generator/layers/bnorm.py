import copy
from common import round_to_simd, generate_param_string, S, fill_tensor, zero_out_tensor
import numpy as np
from conv import block_kernel, block_bias
from scale import set_scale_dim, parse_scale, allocate_scale_lines

def set_bnorm_dim(params, bot_tensor):
    set_scale_dim(params, bot_tensor)

def parse_bnorm(json_param):
    params = parse_scale(json_param)
    params["type"] = "bnorm"
    return params

def allocate_bnorm_lines(lparam):
    return allocate_scale_lines(lparam)

