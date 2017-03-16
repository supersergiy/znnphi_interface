#pragma once

class ZnnPhiConvWrapper {
private:
    void *convEngine;
public:
    ZnnPhiConvWrapper();
    virtual void compute(float *in, float *out, float *ker, float *bi);
};

typedef ZnnPhiConvWrapper* (*CreateConvWrapper_fp)(); 
typedef void (*DestroyConvWrapper_fp)(ZnnPhiConvWrapper*);
