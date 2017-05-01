#pragma once
#include "ZnnPhiConvWrapper.hpp"
//#include <cxx_wrap.hpp>

#define DEFAULT_HT 2
#define DEFAULT_CORES 64 

namespace znn
{
namespace phi
{

class ZnnPhiConvLayer {
private:
    CreateConvWrapper_fp  createConvWrapper;
    DestroyConvWrapper_fp destroyConvWrapper;
    ZnnPhiConvWrapper *conv_wrapper;

public:
    ZnnPhiConvLayer(int bn, int ifm, int ofm, int id,
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
/*
JULIA_CPP_MODULE_BEGIN(registry)
    //using namespace cpp_types;

    cxx_wrap::Module& types = registry.create_module("CppTypes");

    types.add_type<znn::phi::ZnnPhiConvLayer>("ZnnPhiConvLayer")
          .constructor<int, int, int, int,
                       int, int ,int,
                       int, int, int, int>()
    ;
  //.method("forward", &ZnnPhiConvLayer::forwad)
JULIA_CPP_MODULE_END

*/
