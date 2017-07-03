#pragma once

namespace znn
{
namespace phi
{

class ConvLayer {
private:
    void *convEngine;
public:
    ConvLayer();
    virtual ~ConvLayer();
    virtual void compute(float const* __restrict in, float *out, 
                         float const* __restrict ker, 
                         float const* __restrict bi,
                         float const* __restrict scale);
};

typedef ConvLayer* (*CreateConvLayer_fp)(); 
typedef void (*DestroyConvLayer_fp)(ConvLayer*);

}
}
