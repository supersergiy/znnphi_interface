#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<128,1024,1024,1,14,1,3>();
}
