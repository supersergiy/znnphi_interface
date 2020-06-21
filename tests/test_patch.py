#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from pznet.pznet import PZNet
from time import time

# 3.05 sec per patch
#net = PZNet( os.path.expanduser("~/workspace/s1_analysis/13_inference_pznet/s1net/core8" ))


#net = PZNet('/tmp/s1net/core2')
# 2.9 sec
net = PZNet('/tmp/s1net/core8')
#net = PZNet('/tmp/s1net/core16')
#net = PZNet('/tmp/s1net/core32')

input_patch = np.random.rand(20, 256, 256).astype('float32')

start = time()
output_patch = net.forward(input_patch)
print('time per patch: ', time()-start)
