#!/usr/bin/python
import numpy as np
import znet
from operator import mul

in_dim = [1,1,18,192,192]
z = znet.znet()
in_a = np.arange((reduce(mul, in_dim)))
out_a = z.forward(in_a)
print out_a
print out_a.shape
