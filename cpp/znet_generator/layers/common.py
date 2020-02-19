from math import ceil
import numpy as np
import os

def get_simd_width(arch):
    if arch == "AVX2":
        s = 8
    elif arch == "AVX512":
        s = 16
    else:
        raise Exception("Unsupported architecture")
    return s

def round_to_simd(n, arch):
    s = get_simd_width(arch)
    return int(ceil(float(n) / s) * s);

def generate_param_string(allocation_params):
    return ", ".join(["{}"]*len(allocation_params)).format(*allocation_params)

def write_values_to_file(values, file_name):
    with open(file_name, 'w') as f:
        for v in values:
            f.write("{0:.10f} ".format(float(v)))

def fill_tensor(tname, values):
    lines = []

    #TODO: parametrize
    out_directory  = './out/weights'
    data_file_name = '{}.data'.format(tname)
    data_path      = os.path.join(out_directory, data_file_name)
    write_values_to_file(values, data_path)
    fill_in_array = 'readArrayFromFile(tensors["{}"]->data(), weights_path + "{}");'.format(tname, data_file_name)

    lines.append(fill_in_array)

    return lines

def zero_out_tensor(tname):
    lines = []
    lines.append('tensors["{}"]->set_to_const(0);'.format(tname))
    return lines


