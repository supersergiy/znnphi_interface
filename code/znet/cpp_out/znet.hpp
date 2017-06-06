#include <string.h>
#include <znn/tensor/tensor.hpp>
#include <znn/layer/layer.hpp>

namespace znn {
namespace phi {

class Znet {
    //TODO: look into boost ptr_map
    using hbw_map = std::map<std::string, znn::phi::hbw_array<float> *>;
    public:
        hbw_map tensors;
        hbw_map weights;

        std::vector<znn::phi::Layer *> layers;
    public:
        Znet(void);
        ~Znet(void) {
           for (hbw_map::iterator it = tensors.begin(); it != tensors.end(); it++) {
              delete it->second;
           }
           for (hbw_map::iterator it = weights.begin(); it != weights.end(); it++) {
              delete it->second;
           }
        }

        //TODO: add input and output data
        void forward(void);
};

}
}
