#pragma once

namespace znn
{
namespace phi
{

class ZnnPhiConvWrapper {
private:
    void *convEngine;
public:
    ZnnPhiConvWrapper();
    virtual ~ZnnPhiConvWrapper();
    virtual void compute(float const* __restrict in, float *out, 
                         float const* __restrict ker, 
                         float const* __restrict bi);
};

typedef ZnnPhiConvWrapper* (*CreateConvWrapper_fp)(); 
typedef void (*DestroyConvWrapper_fp)(ZnnPhiConvWrapper*);

}
}
