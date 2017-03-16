#pragma once
#include "ZnnPhiConvWrapper.hpp"
#define DEFAULT_HT 2
#define DEFAULT_CORES 32 


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

    void forward(float *in, float *out, float *ker, float *bi);
};
