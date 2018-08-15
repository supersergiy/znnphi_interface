#!/usr/bin/python
import h5py
import numpy as np
import sys
import os
import glob
import pznet

file_spec = sys.argv[1]
test_list = glob.glob(file_spec)

for test_folder in test_list:
   net_path   = os.path.join(test_folder, "net.prototxt")
   weights_path = os.path.join(test_folder, "weights.h5")
   in_path      = os.path.join(test_folder, "in.h5")
   out_path     = os.path.join(test_folder, "out.h5")
   print net_path

   test_name = filter(None, test_folder.split('/'))[-1]
   znet_path = "/opt/znets/{}_reference".format(test_name)
   lib_path  = os.path.join(znet_path, "lib")

   z = pznet.znet()
   z.create_net(net_path, weights_path, znet_path, "AVX2", 2, 2, 0)
   z.load_net(znet_path, lib_path)

   in_data = h5py.File(in_path)["main"][:]
   out_data = z.forward(in_data)

   out_file = h5py.File(out_path)
   out_file.create_dataset('/main', data=out_data)
   out_file.close()
