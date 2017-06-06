#include <znn/tensor/tensor.hpp>
#include <map>
#include <string.h>
int main(void)
{
   std::map<int, znn::phi::hbw_array<float>* > m;
   m[0] =  new znn::phi::hbw_array<float>(20);
   //znn::phi::hbw_array<float> *a = new znn::phi::hbw_array<float>(20);
}

