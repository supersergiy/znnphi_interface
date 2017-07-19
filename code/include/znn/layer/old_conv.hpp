#pragma once
#include <znn/layer/layer.hpp>
#include <znn/jit/jit.hpp>

#define DEFAULT_HT 2 
#define DEFAULT_CORES 2 

namespace znn
{
namespace phi
{

class ConvLayer: public Layer{
   using Layer::forward;

private:
    Layer *conv_layer;

public:
    ConvWrapper(int bn, int ifm, int ofm, int id,
                int ihw, int kd, int khw,
                int padd=0, int padhw=0, bool Activation=false, bool AddOrOverwrite=false,
                int cores=DEFAULT_CORES, int ht=DEFAULT_HT)
    {
       std::stringstream params;

       params << "BN="    << bn    << " ";
       params << "IFM="   << IFM   << " ";
       params << "OFM="   << OFM   << " ";
       params << "ID="    << ID    << " ";
       params << "IHW="   << IHW   << " ";
       params << "KHW="   << KD    << " ";
       params << "PADD="  << PADD  << " ";
       params << "PADHW=" << PADHW << " ";
       params << "CORES=" << cores << " ";
       params << "HT="    << ht    << " ";
       params << "ACTIVATION="     << Activation     << " ";
       params << "ADDOROVERWRITE=" << ADDOROVERWRITE << " ";

       inner_layer = jitLayer("conv", param_string.str());
    }
   
    ~ConvWrapper()
    {
       delete inner_layer; //TODO: this produces memory leak, but let's fix this later
    }

    void forward(float const* __restrict in, float *out,
                 float const* __restrict ker,
                 float const* __restrict bi,
                 float const* __restrict scale)
    {
      inner_layer->forward(in, out, ker, bi, scale);
    }
};

}
}
