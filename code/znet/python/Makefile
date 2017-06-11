
ZNN_INCLUDES = -I/home/ubuntu/znnphi_interface/code/include -I/home/ubuntu/znnphi_interface/code/include/znn -I/home/ubuntu/znnphi_interface/ 

.PHONY: znet
znet:
	icpc -pthread -std=c++14 `python-config --cflags --ldflags` -Ipybind11/include $(ZNN_INCLUDES) -fmax-errors=5 -DZNN_NO_HBW -DZNN_NO_CUDA -DNDEBUG -O3 -ffast-math -fno-omit-frame-pointer -DZNN_AVX -fPIC -lznnphiconv -c znet.cpp znet_numpy_wrapper.cpp 
	icpc znet.o znet_numpy_wrapper.o -lznnphiconv -shared -o znet.so 
	./client_znet.py
