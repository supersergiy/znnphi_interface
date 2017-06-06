#include <znn/tensor/tensor.hpp>
#include <string.h>

#include "layer.hpp"

namespace znn {
namespace phi {

class Znet {
    public:
        std::map<std::String, znn::phi::hbw_array<float> > tensors;
        std::map<std::String, znn::phi::hbw_array<float> > weights;

        std::vector<znn::phi::Layer> layers
    public:
        Znet(void);
        //TODO: add input and output data
        void forward(void);
};

}
}
