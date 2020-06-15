# Compile kaffe model (hdf5 and prototxt files)

set environment variable:
`export ZNNPHI_PATH="your/pznet/path"`

download the network files:
```
wget https://github.com/seung-lab/DeepEM/releases/download/S1/deploy.prototxt
wget https://github.com/seung-lab/DeepEM/releases/download/S1/train_iter_790000.caffemodel.h5
```

install intel parallel package with license.
install anaconda3 and setup environment.
```
conda create -n environment.yml
```

compile network:
```
python scripts/compile_net.py --net deploy.prototxt --weights train_iter_790000.caffemodel.h5 --cores 4 --ht=2 --output-znet-path /tmp/s1net
```


# Old version usage with docker
```
sudo docker run -it  -v /opt/intel/licenses -v $(pwd):/seungmount  seunglab/pznet:devel /opt/znnphi_interface/code/scripts/sergify.py -n /seungmount/deploy.prototxt -w /seungmount/weights.h5 -o /seungmount/out
```

You'll need to change some parameters depending on where your files are stored(docker -v params and sergify.py -n -w and -o)

For more information on `sergify.py` params you can run 

```
sudo docker run -it  seunglab/pznet:devel /opt/znnphi_interface/code/scripts/sergify.py -h
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
- Jingpeng Wu refactored the code to make it more pythonic.
