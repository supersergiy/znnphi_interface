#pragma once
#include "ZnnPhiConvWrapper.hpp"

#define DEFAULT_HT 2
#define DEFAULT_CORES 64 

namespace znn
{
namespace phi
{

class ZnnPhiConvLayer {
private:
    ZnnPhiConvWrapper *conv_wrapper;
    CreateConvWrapper_fp  createConvWrapper;
    DestroyConvWrapper_fp destroyConvWrapper;

public:
    ZnnPhiConvLayer(void);
    /*ZnnPhiConvLayer(int bn, int ifm, int ofm, int id,
                    int ihw, int kd, int khw,
                    int padd=0, int padhw=0,
                    int cores=DEFAULT_CORES, int ht=DEFAULT_HT);*/
   
    void init(int bn, int ifm, int ofm, int id,
                      int ihw, int kd, int khw,
                      int padd=0, int padhw=0,
                      int cores=DEFAULT_CORES, int ht=DEFAULT_HT);
   
    ~ZnnPhiConvLayer(void);

    void forward(float const* __restrict in, float *out,
                 float const* __restrict ker,
                 float const* __restrict bi);
};

}
}
