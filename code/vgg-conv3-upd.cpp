#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<64,128,256,1,58,1,3>();
}
