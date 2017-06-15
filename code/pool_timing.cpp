#include <znn/layer/pool/pool.hpp>
#include "znn/tensor/tensor.hpp"
#include <chrono>
#include <iostream>
using namespace znn::phi;

int main()
{
   int bn=1;
   int fm=32;
   int id=18;
   int ihw=192;
   MaxPoolingLayer mpl(bn, fm, id, ihw, 1, 2, 1, 2);
   
   hbw_array<float> a(bn*fm*id*ihw*ihw);
   hbw_array<float> b(bn*fm*id*ihw*ihw);

   auto begin = std::chrono::high_resolution_clock::now();
   int iters = 100;
   for (int i = 0; i < iters; i++) {
      mpl.forward(a.data(), b.data(), NULL, NULL);
   }

   auto end = std::chrono::high_resolution_clock::now();
   auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
      .count();

   double secs = static_cast<double>(duration) / 1000000;

   std::cout << "Secs   : " << (secs/iters) << "\n";
}
