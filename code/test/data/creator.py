#!/usr/bin/python
import h5py
import numpy as np
import sys

FILENAME = sys.argv[1] + ".h5"

f = h5py.File(FILENAME, "w")
dset = f.create_dataset("data", (1,8,18,192,192), dtype='f')

