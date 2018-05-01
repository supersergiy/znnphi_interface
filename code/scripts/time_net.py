#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys
from time import time
from optparse import OptionParser

def timing_test(base, N, recompile, architecture, core_options, cpu_offset, optimization, ignore, time_each, input_mode, run):
    test_name = list(filter(None, base.split('/')))[-1]

    net_path = os.path.join(base, "net.prototxt")
    weights_path = os.path.join(base, "weights.h5")
    input_path =  os.path.join(base, "in.h5")

    z = pznet.znet()
    znet_path = "/opt/znets/{}_{}cores_{}".format(test_name, core_options["conv"][0], optimization)
    lib_path  = "/opt/znets/lib"#os.path.join(znet_path, "lib")
    if recompile:
        print ("Recompiling...")
        z.create_net(net_path, weights_path, znet_path, architecture, core_options, cpu_offset,
                    opt_mode=optimization, ignore=ignore, time_each=time_each)

    if run:
        print ("Loading net...")
        z.load_net(znet_path, lib_path)

        in_shape = z.get_in_shape()
        if input_mode == 'read':
            in_file  = h5py.File(input_path)
            in_a     = in_file["main"][:]


        print ("Running net...")
        durations = []
        for i in range(N):
            if input_mode == 'random':
               in_a = np.random.random_sample(in_shape)
            elif input_mode == 'zero':
               in_a = np.zeroes(in_shape)
            elif input_mode == 'one':
               in_a = np.ones(in_shape)
            s = time()
            z.forward(in_a)
            e = time()
            durations.append(e-s)
        return sum(durations) / len(durations)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-b", "--base", dest="base_path")
    parser.add_option("-i", "--input_mode", dest="input_mode", default="random")
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
    parser.add_option("--time_each", action="store_true", dest="time_each", default=False)
    parser.add_option("--dont_run", action="store_false", dest="run", default=True)
    parser.add_option("--ignore", action="append", dest="ignore", default=["ignore"])

    parser.add_option("--arch", dest="architecture", default="AVX2",
            help="The cpu architexture: {AVX2, AVX512}")
    (options, args) = parser.parse_args()

    cpu_offset   = 0
    architecture = options.architecture
    base         = options.base_path
    recompile    = options.recompile
    optimization = options.optimization
    input_mode   = options.input_mode
    N            = options.num_iter
    ignore       = ','.join(options.ignore)
    time_each    = options.time_each
    core_options = {}
    core_options["conv"] = [options.conv_cores, options.conv_ht]
    core_options["act"]  = [options.act_cores, options.act_ht]
    core_options["lin"]  = [options.lin_cores, options.lin_ht]

    average_time = timing_test(base=base, N=N, recompile=recompile, architecture=architecture, cpu_offset=cpu_offset, core_options=core_options,
                               optimization=optimization, ignore=ignore, time_each=time_each, input_mode=input_mode, run=options.run)
    if not options.output_path is None:
        of = file(options.output_path, 'a')
        of.write("{} {} {}: {}\n".format(architecture, base, optimization, average_time))
    else:
        print (average_time)
