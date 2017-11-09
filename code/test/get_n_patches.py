#!/usr/bin/python
import h5py
import numpy as np
import sys
from random import randint

n = int(sys.argv[1])
patch_z  = int(sys.argv[2])
patch_xy = int(sys.argv[3])

in_path      = "/home/ubuntu/inference_data/data/golden_chunk.h5"
out_path     = "./{}patches.h5".format(n)

in_data  = h5py.File(in_path)["main"][...]
out_data = np.zeros((n, 1, patch_z, patch_xy, patch_xy))
in_shape = in_data.shape

for i in range(n):
    x = randint(0, in_shape[-1] - patch_xy)
    y = randint(0, in_shape[-2] - patch_xy)
    z = randint(0, in_shape[-3] - patch_z)

    out_data[i] = in_data[z:z+patch_z, y:y+patch_xy, x:x+patch_xy] / 256.0

out_file = h5py.File(out_path)
out_file.create_dataset('/main', data=out_data)
out_file.close()
