#include "znn/intrin.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/layer/block_unblock/block_data_layer.hpp"
#include "znn/layer/block_unblock/unblock_data_layer.hpp"

#include <chrono>
#include <iostream>
#include <string>

using namespace znn::phi;

void testCancellation(size_t b, size_t fm, size_t d, size_t hw)
{
    block_data_layer   bl(b, fm, hw, d);
    unblock_data_layer ubl(b, fm, hw, d);
    
    const int SIZE=b*fm*hw*hw*d;
    hbw_array<float> in(seq_init, SIZE);
    hbw_array<float> out1(one_init, SIZE);
    hbw_array<float> out2(one_init, SIZE);
    std::cout << "SIMD_WIDTH: " << SIMD_WIDTH << std::endl;
    bl.execute(in.data(), out1.data());
    ubl.execute(out1.data(), out2.data());
    for (int i = 0; i < SIZE; i++) {
        std::cout << in.data()[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << out1.data()[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << out2.data()[i] << " ";
    }
    std::cout << std::endl;
    assert(memcmp(in.data(), out2.data(), SIZE*sizeof(float)));
}

int main(void)
{
    testCancellation(1,16,1,2);
}


