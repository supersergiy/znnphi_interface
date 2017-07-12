#include <znn/layer/layer_templates.hpp>
#include <znn/layer/layer.hpp>

namespace znn {
namespace phi {

typedef convTemplate<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, true, true> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
    return reinterpret_cast<Layer*>(new parametrized_template_layer());
}

}
}
