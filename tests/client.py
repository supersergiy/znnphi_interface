#!/usr/bin/python
import sys
sys.path.append("./out/")
import numpy as np
import znet
from operator import mul

in_dim = [1,1,18,192,192]
z = znet.znet("./out/weights/")
in_a = np.ones((reduce(mul, in_dim)), dtype=np.float)
out_a = z.forward(in_a)
print out_a
print out_a.shape
