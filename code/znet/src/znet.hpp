#include <string.h>
#include <znn/tensor/tensor.hpp>
#include <znn/layer/layer.hpp>
#include <vector>

namespace znn {
namespace phi {

class Znet {
    //TODO: look into boost ptr_map
    using hbw_map   = std::map<std::string, znn::phi::hbw_array<float> *>;
    using layer_map = std::map<std::string, znn::phi::Layer *>;
    public:
        hbw_map tensors;
        
        layer_map layers;
        std::vector<std::string> layer_order;
        
        size_t input_size;
        std::vector<size_t> out_strides; 
        std::vector<size_t> out_shape;
        size_t out_dim = 1;

    public:
        Znet() {};
        Znet(std::string);
        ~Znet(void) {
           //std::cout << "in destructor\n";
           for (hbw_map::iterator it = tensors.begin(); it != tensors.end(); it++) {
              delete it->second;
           }
           //TODO: there's a memory leak here
        }

        //TODO: add input and output data
        void forward(void);
};

}
}
