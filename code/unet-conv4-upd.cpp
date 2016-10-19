#include "znn/bench/update.hpp"

using namespace znn::phi;

int main()
{
    benchmark_update<1,512,512,1,66,1,3>();
}
