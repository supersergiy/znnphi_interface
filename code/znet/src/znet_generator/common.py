from math import ceil
import numpy as np

#TODO: FIX FOR VARIABLE SIMD
S = 8

def round_to_simd(n):
    return int(ceil(float(n) / S) * S);

def block_kernel(kernel, lparam):
    kdim = lparam["kernel_dim"]
    kernel = kernel.reshape(kdim)
    blocked_kernel = np.array([0.0]*lparam['kernel_size'])

    def h5ker_to_znnphiker(ofm, ifm, kz, kx, ky):
        total_ofms = round_to_simd(kdim[0])
        total_ifms = round_to_simd(kdim[1])

        offset = ofm/S
        offset *= total_ifms/S
        offset += ifm/S
        offset *= kdim[2]
        offset += kz
        offset *= kdim[3]
        offset += kx
        offset *= kdim[4]
        offset += ky
        offset *= S
        offset += ifm % S
        offset *= S
        offset += ofm % S
        return offset

    # h5 weight format: ofm-ifm-kz-kx-ky
    # output format: ofm/S-ifm/S-kz-kx-ky-ofm%S-ifm%S
    for ofm in range(kdim[0]):
        for ifm in range(kdim[1]):
            for kz in range(kdim[2]):
                for kx in range(kdim[3]):
                    for ky in range(kdim[4]):
                        znnphi_index = h5ker_to_znnphiker(ofm, ifm, kz, kx, ky)
                        blocked_kernel[znnphi_index] = kernel[ofm][ifm][kz][kx][ky]

    return blocked_kernel


def block_bias(bias, lparam):
    blocked_bias = np.array([0.0]*lparam['bias_size'])
    for ofm in range(lparam["ofm"]):
        blocked_bias[ofm] = bias[ofm]

    return blocked_bias

