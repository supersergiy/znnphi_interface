
#include <iostream>
#include <znn/jit/jit.hpp>
#include <znn/layer/layer.hpp>
#include <znn/tensor/tensor.hpp>

int main()
{
   std::string params = "BN=1 IFM=1 OFM=28 ID=18 IHW=196 KD=1 KHW=5 OUT_PADD=0 OUT_PADHW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2";
   znn::phi::Layer* l=znn::phi::jitMakeLayer("conv", params);
   znn::phi::hbw_array<float> a(5531904), b(21678336), c(6400), d(32), e(32);
   std::cout << "Running 1...!\n";
   l->forward(a.data(), b.data(), c.data(), d.data(), NULL); 
   std::cout << "Done!\n";
}
