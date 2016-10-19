#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<1,64,64,1,570,1,3>();
}
