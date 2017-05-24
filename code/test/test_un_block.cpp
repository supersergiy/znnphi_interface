#include "znn/intrin.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/layer/block_unblock/block_data_layer.hpp"
#include "znn/layer/block_unblock/unblock_data_layer.hpp"

#include <chrono>
#include <iostream>
#include <string>

namespace znn
{
namespace phi
{

void testCancellation(size_t b, size_t fm, size_t hw, size_t d)
{
    block_data_layer   bl(b, fm, hw, d);
    unblock_data_layer ubl(b, fm, hw, d);
    
    const int SIZE=b*fm*hw*hw*d;
    hbw_array<float> in(rand_init, SIZE);
    hbw_array<float> out1(rand_init, SIZE);
    hbw_array<float> out2(one_init, SIZE);

    bl.execute(in.data(), out1.data());
    ubl.execute(out1.data(), out2.data());
    assert(memcmp(in.data(), out2.data(), SIZE*sizeof(float)));
}

int main(void)
{
    testCancellation(1,16,1,1);
}

}
}


