#!/usr/bin/python
import pznet
import os
import numpy as np
import h5py
import sys

cores = 2
ht    = 2
cpu_offset = 2

base = sys.argv[1]
create = True
if len(sys.argv) > 2:
    create = False

test_name = filter(None, base.split('/'))[-1]

net_path = os.path.join(base, "net.prototxt")
weights_path = os.path.join(base, "weights.h5")
input_path =  os.path.join(base, "in.h5")
reference_path =  os.path.join(base, "out.h5")

in_file  = h5py.File(input_path)
in_a     = in_file["main"][:]
znet_path = "/opt/znets/{}_{}cores".format(test_name, cores)
lib_path  = os.path.join(znet_path, "lib")
z = pznet.znet()

if create:
    print "Creating net..."
    z.create_net(net_path, weights_path, znet_path, cores, ht, cpu_offset)
print "Running net..."
z.load_net(znet_path, lib_path)

for i in range(1):
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
    max_rel_d = max_d / fr[i]
    print "Max rel d: {}".format(max_rel_d)
    print "Max d: {}".format(max_d)
    print "Average d: {}".format(np.average(rel_d))

    if np.isnan(error):
        print "Not congrats! Error == {}".format(error)
        import pdb; pdb.set_trace()
    elif error > 0.010:
        print "Not congrats! Error == {}".format(error)
        import pdb; pdb.set_trace()
    else:
        print "Congrats! All pass. Error == {}".format(error)
        import pdb; pdb.set_trace()
#out_file = h5py.File(output_path)
#out_file.create_dataset("data", data=out_a)

