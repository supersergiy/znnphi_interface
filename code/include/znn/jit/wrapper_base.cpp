#include <znn/layer/layer_templates.hpp>
#include <znn/layer/layer.hpp>

namespace znn {
namespace phi {

typedef [LAYER_NAME]Template<[TEMPLATE_PARAMS]> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
    return reinterpret_cast<Layer*>(new parametrized_template_layer());
}

}
}
