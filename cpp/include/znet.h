#include <string.h>
#include "znn/tensor/tensor.hpp"
#include "znn/layer/layer.hpp"
#include <vector>

namespace znn {
namespace phi {

class Znet {
    //TODO: look into boost ptr_map
    using hbw_map   = std::map<std::string, znn::phi::hbw_array<float> *>;
    using layer_map = std::map<std::string, znn::phi::Layer *>;

    public:
        std::string lib_path; //sets the place where pznet is gonna be looking for the layer .so files

        hbw_map tensors; // maps tensor names to tensor objects
        layer_map layers; //maps layer names to layer objects

        std::vector<std::string> layer_order;
        
        std::vector<size_t> out_strides; //strides for each output dim
        // don't need input strides because the input layout is standard
        std::vector<size_t> out_shape;   //output shape
        std::vector<size_t> in_shape;    //input shape
        size_t out_dim = 5; //output is usually 5-dimensional
        size_t in_dim  = 5; //input is usually 5-dimensional

        size_t input_size; //input size in number of elements(!) 
    public:
        Znet() {};
        Znet(std::string weights_path, std::string lib_path);

        ~Znet(void) {
           for (hbw_map::iterator it = tensors.begin(); it != tensors.end(); it++) {
              delete it->second;
           }
           //TODO: there's a memory leak here, but it segfaults when I try to fix
        }

        //TODO: add input and output data
        void forward(void);
};

}
}
