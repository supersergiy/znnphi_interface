#include "params.hpp"
#include "ZnnPhiConvEngine.hpp"
#include "ZnnPhiConvWrapper.hpp"

typedef ZnnPhiConvEngine<Cores_v, HT_v, B_v, 
                         IFM_v, OFM_v, ID_v, 
                         IHW_v, KD_v, KHW_v, 
                         PADD_v, PADHW_v    >    parametrizedConvEngine;

extern "C" void destroyConvWrapper(ZnnPhiConvWrapper *object)
{
    delete object;
}

extern "C" ZnnPhiConvWrapper* createConvWrapper(void)
{
    return new ZnnPhiConvWrapper(); 
}

ZnnPhiConvWrapper::ZnnPhiConvWrapper()
{
    convEngine = new parametrizedConvEngine();
}

void ZnnPhiConvWrapper::compute(float *in, float *out, float *ker, float *bi)
{
    reinterpret_cast<parametrizedConvEngine>(convEngine)->compute();
}

