#pragma once
#include "ZnnPhiConvWrapper.hpp"
#include <boost/python.hpp>
#define DEFAULT_HT 2
#define DEFAULT_CORES 32

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
    ZnnPhiConvLayer(void) {}
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

char const* greet()
{
       return "hello, world";

}

BOOST_PYTHON_MODULE(libznnphiconv)
{
    using namespace boost::python;
    // Create the Python type object for our extension class and define __init__ function.
    class_<znn::phi::ZnnPhiConvLayer>("Conv", init<int, int ,int,
                                                         int, int, int,
                                                         int, int, int,
                                                         int, int>())
        //.def("greet", &hello::greet)  // Add a regular member function.
        //.def("invite", invite)  // Add invite() as a regular function to the module.
    ;

    //def("greet", greet); // Even better, invite() can also be made a member of module!!!
}
