#pragma once

#include "znn/layer/conv/propagation/executable.hpp"
#include "znn/layer/conv/propagation/full_layer/problem.hpp"
#include "znn/layer/conv/propagation/full_layer/schedule.hpp"
#include "znn/util/kernel_launcher.hpp"
#include "znn/tensor/tensor.hpp"
#include <functional>
#include <vector>

namespace znn
{
namespace phi
{
namespace propagation
{

template <long_t Threads, class P, bool Activation, bool AddToOutput>
struct full_layer
{
private:
    using full_sub_problem =
        sub_problem_t<0, P::batch_size, 0, P::ofm_len, 0, P::image_d::size, 0,
                      P::image_h::size, 0, P::image_w::size>;

    using problem = problem_t<Threads, P, full_sub_problem, Activation, AddToOutput>;

    kernel_launcher*                   launcher;
    std::vector<std::function<void()>> fns;
    exec_vector                        executables;

    float const* in_;
    float*       out_;
    float const* kernels_;
    float const* biases_;
    float const* scale_;
    hbw_array<float> *add_to_; 

public:
    full_layer(kernel_launcher* l)
        : launcher(l)
        , fns(Threads)
        , executables(Threads)
    {
        scheduler<problem>::schedule(0, executables);
        for (long_t i = 0; i < Threads; ++i)
        {
            fns[i] = [i, this]() {
                for (auto const& e : this->executables[i])
                {
                    if (AddToOutput) {
                       e(this->in_, this->add_to_->data(), this->kernels_, this->biases_, this->scale_, this->out_);
                    }
                    else {
                       e(this->in_, this->out_, this->kernels_, this->biases_, this->scale_, NULL);
                    }
                }
            };
        }
    }

    ~full_layer() {
       delete add_to_;
    }
   
    void execute(float const* __restrict i, float* __restrict o,
                 float const* __restrict k, float const* __restrict b,
                 float const* __restrict s)
    {
        in_      = i;
        out_     = o;
        kernels_ = k;
        biases_  = b;
        scale_   = s;
        if (AddToOutput) {
           add_to_ = new hbw_array<float>(zero_init, P::batch_size * P::out_batch_stride);
           std::cout << P::batch_size * P::out_batch_stride << std::endl;
        }
        else {
           add_to_ = NULL;
        }

        launcher->launch(&(fns[0]));
    }

    long_t flops() const
    {
        return P::batch_size * P::ifm_len * P::ofm_len * SIMD_WIDTH *
               SIMD_WIDTH * P::image_d::size * P::image_h::size *
               P::image_w::size * P::conv_d::size * P::conv_h::size *
               P::conv_w::size * 2;
    }
};

} // namespace propagation
} // namespace phi
} // namespace znn
