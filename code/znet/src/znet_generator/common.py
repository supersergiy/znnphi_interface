from math import ceil
import numpy as np
from layers import round_to_simd
#TODO: FIX FOR VARIABLE SIMD

def generate_params_string(allocation_params):
    return ", ".join(["{}"]*len(allocation_params)).format(allocation_params)
