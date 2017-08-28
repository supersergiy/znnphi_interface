#!/usr/bin/python
import h5py
import numpy as np

in_path      = "/sergiy_shared/overlap/img.h5"
out_path     = "/sergiy_shared/znnphi_interface/code/znet/reference/patch.h5"

in_data = h5py.File(in_path)["main"][...]

out_data = in_data[0:18, 0:192, 0:192] / 256.0

out_file = h5py.File(out_path)
out_file.create_dataset('/main', data=out_data)
out_file.close()
