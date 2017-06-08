from math import ceil
#TODO: FIX FOR VARIABLE SIMD
S = 8
def round_to_simd(n):
    return int(ceil(float(n) / S) * S);
