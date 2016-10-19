#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<1,256,256,1,138,1,3>();
}
