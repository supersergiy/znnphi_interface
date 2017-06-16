from math import ceil
import numpy as np

#TODO: FIX FOR VARIABLE SIMD
S = 8

def round_to_simd(n):
    return int(ceil(float(n) / S) * S);

def generate_params_string(allocation_params):
    return ", ".join(["{}"]*len(allocation_params)).format(allocation_params)
