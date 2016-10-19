#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<1,128,128,1,282,1,3>();
}
