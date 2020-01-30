#include <znn/layer/layer_templates.hpp>
#include <znn/layer/layer.hpp>

namespace znn {
namespace phi {

typedef ConvTemplate<1, 2, 1, 144, 104, 22, 14, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2, true, 0> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
   Layer *result = reinterpret_cast<Layer*>(new parametrized_template_layer());
   return result;
}

}
}
