#include <znn/layers/layers.hpp>

namespace znn {
namespace phi {

typedef convTemplate<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, true, true> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
    return parametrized_template_layer();
}

}
}
