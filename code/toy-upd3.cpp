#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<32,256,256,10,25,3,3>();
}
