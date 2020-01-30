#include <znn/layer/layer_templates.hpp>
#include <znn/layer/layer.hpp>

namespace znn {
namespace phi {

typedef ConvTemplate<2, 2, 1, 48, 48, 18, 50, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, false, 0> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
   Layer *result = reinterpret_cast<Layer*>(new parametrized_template_layer());
   return result;
}

}
}
