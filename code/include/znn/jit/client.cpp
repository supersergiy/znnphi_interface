#include <znn/layer/layer_templates.hpp>
#include <znn/layer/layer.hpp>
#include <znn/tensor/tensor.hpp>

namespace znn {
namespace phi {

typedef ConvTemplate<1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, true, true> parametrized_template_layer;

extern "C" Layer* createLayer(void)
{
    Layer* result = (new parametrized_template_layer());
    return result;
}


}
}




int main()
{
   znn::phi::Layer *l = znn::phi::createLayer();
   znn::phi::hbw_array<float> a(100), b(100), c(100), d(100), e(100);
   l->flops();
   (l)->forward(a.data(), b.data(), c.data(), d.data(), e.data());
   return 0;
}
