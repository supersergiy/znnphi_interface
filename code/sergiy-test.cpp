#include "znn/bench/forward3.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<32, 32, 32, 18, 192, 1, 3>("conv1");
    //benchmark_forward<8, 80, 80, 300, 300, 3, 3>("conv2");
    //benchmark_forward<64, 80, 80, 192, 192, 3, 3>("conv3");
    //benchmark_forward<1, 64, 64, 570, 570, 9, 1>("conv4");
    //benchmark_forward<1, 8, 8, 1, 3, 1, 2>("conv1b");
    //benchmark_forward<1, 128, 128, 1, 282, 1, 3>("conv2b");
    //benchmark_forward<1, 256, 256, 1, 138, 1, 3>("conv3b");
    //benchmark_forward<1, 512, 512, 1, 66, 1, 3>("conv4b");
    //benchmark_forward<1, 1024, 1024, 1, 30, 1, 3>("conv5b");
}
