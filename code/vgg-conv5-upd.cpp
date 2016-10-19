#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<64,256,512,1,30,1,3>();
}
