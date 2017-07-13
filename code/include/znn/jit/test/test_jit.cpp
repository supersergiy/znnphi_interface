
#include <iostream>
#include <znn/jit/jit.hpp>
#include <znn/layer/layer.hpp>
#include <znn/tensor/tensor.hpp>

int main()
{
   std::string params = "CORES=1 HT=1 BN=1 IFM=1 OFM=1 ID=1 IHW=2 KD=1 KHW=1 PADD=1 PADHW=1 Activation=0 AddOrOverwrite=0";
   znn::phi::Layer* l=znn::phi::jitMakeLayer("conv", params);
   znn::phi::hbw_array<float> a(100000000), b(1000000000), c(100000), d(100000), e(10000);
   std::cout << "Running...!\n";
   l->forward(a.data(), b.data(), c.data(), d.data(), NULL);
   std::cout << "Done!\n";
}
