#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_single_forward<64,1,1  ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,2  ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,3  ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,8  ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,16 ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,32 ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,64 ,16,16,1,128,1,3>();
    benchmark_single_forward<64,1,128,16,16,1,128,1,3>();


    benchmark_single_forward<64,1,1  ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,2  ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,3  ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,8  ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,16 ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,32 ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,64 ,32,32,1,128,1,3>();
    benchmark_single_forward<64,1,128,32,32,1,128,1,3>();


    benchmark_single_forward<64,1,1  ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,2  ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,3  ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,8  ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,16 ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,32 ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,64 ,64,64,1,128,1,3>();
    benchmark_single_forward<64,1,128,64,64,1,128,1,3>();


    benchmark_single_forward<64,1,1  ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,2  ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,3  ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,8  ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,16 ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,32 ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,64 ,128,128,1,128,1,3>();
    benchmark_single_forward<64,1,128,128,128,1,128,1,3>();


    benchmark_single_forward<64,1,1  ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,2  ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,3  ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,8  ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,16 ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,32 ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,64 ,256,256,1,128,1,3>();
    benchmark_single_forward<64,1,128,256,256,1,128,1,3>();

}
