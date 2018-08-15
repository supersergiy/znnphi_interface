#!/usr/bin/python3
import pznet
import os
import numpy as np
import h5py
import sys
from time import time
from optparse import OptionParser

def correctness_test(base, N, recompile, architecture, core_options, cpu_offset, optimization):
    test_name = list(filter(None, base.split('/')))[-1]

    net_path = os.path.join(base, "net.prototxt")
    weights_path = os.path.join(base, "weights.h5")
    input_path =  os.path.join(base, "in.h5")
    reference_path =  os.path.join(base, "out.h5")

    z = pznet.znet()
    znet_path = "/opt/znets/{}_{}cores_{}".format(test_name, core_options["conv"][0], optimization)
    lib_path  = "/opt/znets/lib"#os.path.join(znet_path, "lib")
    if recompile:
        print ("Recompiling...")
        z.create_net(net_path, weights_path, znet_path, architecture, core_options, cpu_offset,
                    opt_mode=optimization)

    print ("Loading net...")
    z.load_net(znet_path, lib_path)

    in_file  = h5py.File(input_path)
    in_a     = in_file["main"][:]

    print ("Running net...")
    out_a    = z.forward(in_a)

    reference_file = h5py.File(reference_path)
    reference_a = reference_file["main"][:]
    np.set_printoptions(precision=2)

    diff_a = reference_a - out_a
    rel_d = np.abs(diff_a) / (out_a + 0.0000000001)
    mask1 = diff_a > 1e-5
    mask2 = rel_d > 1e-5

    #fd = diff_a.flatten()
    #fo = out_a.flatten()
    fr = reference_a.flatten()
    #diffs0 = [np.sum(np.abs(diff_a[0][0][i])) for i in range(18)]
    #diffs1 = [np.sum(np.abs(diff_a[1][0][i])) for i in range(18)]
    errors = rel_d * mask1 * mask2 * reference_a
    error = np.sum((errors*10)**2)

    max_d = np.max(np.abs(diff_a))
    #i = np.argmax(np.abs(diff_a.flatten()))
    max_rel_d = max_d / fr
    print ("Max rel d: {}".format(max_rel_d))
    print ("Max d: {}".format(max_d))
    print ("Average d: {}".format(np.average(rel_d)))

    if np.isnan(error):
        print ("Not congrats! Error == {}".format(error))
        import pdb; pdb.set_trace()
    elif error > 0.010:
        print ("Not congrats! Error == {}".format(error))
        import pdb; pdb.set_trace()
    else:
        print ("Congrats! All pass. Error == {}".format(error))


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-b", "--base", dest="base_path")
    parser.add_option("-o", "--output_path", dest="output_path", default=None)
    parser.add_option("--iter", dest="num_iter", default=20, type="int")
    parser.add_option("-O", dest="optimization", default="full_opt")
    parser.add_option("-c", "--cores", dest="conv_cores", default=2)
    parser.add_option("--ht", dest="conv_ht", default=2)
    parser.add_option("--act_cores", dest="act_cores", default=-1)
    parser.add_option("--act_ht", dest="act_ht", default=-1)
    parser.add_option("--lin_cores", dest="lin_cores", default=-1)
    parser.add_option("--lin_ht", dest="lin_ht", default=-1)
    parser.add_option("--recompile", action="store_true", dest="recompile", default=False)

    parser.add_option("--arch", dest="architecture", default="AVX2",
            help="The cpu architexture: {AVX2, AVX512}")
    (options, args) = parser.parse_args()

    cpu_offset   = 0
    architecture = options.architecture
    base         = options.base_path
    recompile    = options.recompile
    optimization = options.optimization
    N            = options.num_iter
    core_options = {}
    core_options["conv"] = [options.conv_cores, options.conv_ht]
    core_options["act"]  = [options.act_cores, options.act_ht]
    core_options["lin"]  = [options.lin_cores, options.lin_ht]

    average_time = correctness_test(base=base, N=N, recompile=recompile, architecture=architecture, cpu_offset=cpu_offset, core_options=core_options,
                               optimization=optimization)
