[![Docker Hub](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/seunglab/pznet)

# Compile kaffe model (hdf5 and prototxt files)

## Setup Locally
We currently support Ubuntu >=16.04 with Python 3.8.

set environment variable:
`export ZNNPHI_PATH="your/pznet/path"`

download the network files:
```
wget https://github.com/seung-lab/DeepEM/releases/download/S1/deploy.prototxt
wget https://github.com/seung-lab/DeepEM/releases/download/S1/train_iter_790000.caffemodel.h5
```

install intel parallel package with license.
install required packages
```
pip install -r requirements.txt
```

compile network:
```
python scripts/compile_net.py --net deploy.prototxt --weights train_iter_790000.caffemodel.h5 --cores 4 --ht=2 --output-znet-path /tmp/s1net
```

## Run convolutional network inference using compiled net

```python
import numpy as np
from pznet.pznet import PZNet
net = PZNet('my/compiled/net')
input_patch = np.random.rand(20, 256, 256).astype('float32') 
output_patch = net.forward(input_patch)                      
```

# Use Docker Image
We have set up continuous integration and the Docker image will be automatically built with every GitHub commit in the master branch. You can find the Docker image [here](https://hub.docker.com/repository/docker/seunglab/pznet).

You need to mount your local intel directory in order to compile the net. You do not need this for inference, all the required shared libraries are already included.
```
docker run -it -v /opt/intel:/opt/intel seunglab:pznet bash
```


# Citation

```
@inproceedings{popovych2019pznet,
  title={Pznet: Efficient 3d convnet inference on manycore cpus},
  author={Popovych, Sergiy and Buniatyan, Davit and Zlateski, Aleksandar and Li, Kai and Seung, H Sebastian},
  booktitle={Science and Information Conference},
  pages={369--383},
  year={2019},
  organization={Springer}
}

@inproceedings{zlateski2017compile,
  title={Compile-time optimized and statically scheduled ND convnet primitives for multi-core and many-core (Xeon Phi) CPUs},
  author={Zlateski, Aleksandar and Seung, H Sebastian},
  booktitle={Proceedings of the International Conference on Supercomputing},
  pages={1--10},
  year={2017}
}
```

# Credit
- Aleksandar Zlateski implemented the first version of C++ backend. 
- Sergiy Popovych built a python package to generate C++ code and compilation scripts for pratical deployment. It imports caffe models for compilation.
- Jingpeng Wu refactored the code to make it more pythonic with continuous integration.
