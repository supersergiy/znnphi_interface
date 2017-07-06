#pragma once
#include <iostream>

namespace znn {
namespace phi {

class Layer {
    public:
        virtual void forward(float const* __restrict in, float *out,
                             float const* __restrict ker,
                             float const* __restrict bi) 
        {
            std::cerr << "Not implemented" << std::endl;
            exit(EXIT_FAILURE);
        }; 

        virtual void forward(float const* __restrict, float *,
                             float const* __restrict,
                             float const* __restrict,
                             float const* __restrict) 
        {
            std::cerr << "Not implemented" << std::endl;
            exit(EXIT_FAILURE);
        };

        virtual ~Layer() {};
};

}
}
