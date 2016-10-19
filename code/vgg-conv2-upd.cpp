#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<64,64,128,1,114,1,3>();
}
