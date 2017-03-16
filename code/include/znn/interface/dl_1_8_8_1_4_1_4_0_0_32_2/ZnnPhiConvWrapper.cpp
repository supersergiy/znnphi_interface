#include "params.hpp"
#include "znn/interface/ZnnPhiConvEngine.hpp"
#include "znn/interface/ZnnPhiConvWrapper.hpp"

namespace znn 
{
namespace phi
{

typedef ZnnPhiConvEngine<Cores_v, HT_v, B_v, 
                         IFM_v, OFM_v, ID_v, 
                         IHW_v, KD_v, KHW_v, 
                         PADD_v, PADHW_v    >    parametrizedConvEngine;

ZnnPhiConvWrapper::~ZnnPhiConvWrapper()
{
}

ZnnPhiConvWrapper::ZnnPhiConvWrapper()
{

    convEngine = new parametrizedConvEngine();
}

void ZnnPhiConvWrapper::compute(float *in, float *out, float *ker, float *bi)
{
    reinterpret_cast<parametrizedConvEngine*>(convEngine)->compute(in, out, ker, bi);
}

}
}

extern "C" void destroyConvWrapper(znn::phi::ZnnPhiConvWrapper *object)
{
    delete object;
}

extern "C" znn::phi::ZnnPhiConvWrapper* createConvWrapper(void)
{
    return new znn::phi::ZnnPhiConvWrapper(); 
}


