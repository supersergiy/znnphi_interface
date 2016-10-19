#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<64,8,64,1,226,1,3>();
}
