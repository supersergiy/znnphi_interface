#!/usr/bin/python
import h5py
import numpy as np
import sys
from random import randint

n = int(sys.argv[1])

in_path      = "/home/ubuntu/inference_data/img.h5"
out_path     = "./{}patches.h5".format(n)

in_data  = h5py.File(in_path)["main"][...]
out_data = np.zeros((n, 1, 18, 192, 192))
in_shape = in_data.shape

for i in range(n):
    x = randint(0, in_shape[-1] - 192)
    y = randint(0, in_shape[-2] - 192)
    z = randint(0, in_shape[-3] - 18)

    out_data[i] = in_data[z:z+18, y:y+192, x:x+192] / 256.0

out_file = h5py.File(out_path)
out_file.create_dataset('/main', data=out_data)
out_file.close()
