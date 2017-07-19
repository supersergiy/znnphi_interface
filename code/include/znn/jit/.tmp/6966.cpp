#include <znn/layer/layer_templates.hpp>
#include <znn/layer/layer.hpp>

namespace znn {
namespace phi {

typedef ConvTemplate<2, 2, 1, 1, 28, 18, 196, 1, 5, 0, 1, true, true> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
   Layer *result = reinterpret_cast<Layer*>(new parametrized_template_layer());
   return result;
}

}
}
