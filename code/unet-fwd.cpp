#include "znn/bench/forward2.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1, 64, 64, 1, 570, 1, 3>("conv1b");
    // benchmark_forward<1, 128, 128, 1, 282, 1, 3>("conv2b");
    // benchmark_forward<1, 256, 256, 1, 138, 1, 3>("conv3b");
    // benchmark_forward<1, 512, 512, 1, 66, 1, 3>("conv4b");
    // benchmark_forward<1, 1024, 1024, 1, 30, 1, 3>("conv5b");
}
