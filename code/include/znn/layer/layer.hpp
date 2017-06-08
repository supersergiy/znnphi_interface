#pragma once
namespace znn {
namespace phi {

class Layer {
    public:
        virtual void forward(float const* __restrict in, float *out,
                  float const* __restrict ker,
                  float const* __restrict bi){}; //TODO: remove this dummy
        virtual ~Layer(){};
};

}
}
