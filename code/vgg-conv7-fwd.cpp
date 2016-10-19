#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<64,512,512,1,16,1,3>();
}
