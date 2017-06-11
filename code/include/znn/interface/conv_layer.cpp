#include "znn/interface/conv_engine.hpp"
#include "znn/interface/conv_layer.hpp"

namespace znn 
{
namespace phi
{

typedef ConvEngine<Cores_v, HT_v, BN_v, 
                   IFM_v, OFM_v, ID_v, 
                   IHW_v, KD_v, KHW_v, 
                   PADD_v, PADHW_v    >    parametrizedConvEngine;

ConvLayer::~ConvLayer()
{
}

ConvLayer::ConvLayer()
{

    convEngine = new parametrizedConvEngine();
}

void ConvLayer::compute(float const* __restrict in, float *out, 
                        float const* __restrict ker, 
                        float const* __restrict bi)
{
    reinterpret_cast<parametrizedConvEngine*>(convEngine)->compute(in, out, ker, bi);
}

}
}

extern "C" void destroyConvLayer(znn::phi::ConvLayer *object)
{
    delete object;
}

extern "C" znn::phi::ConvLayer* createConvLayer(void)
{
    return new znn::phi::ConvLayer(); 
}


