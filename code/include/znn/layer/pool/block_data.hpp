#include <cstring>
#include <assert.h>
#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct BlockDataLayer: public Layer{
private:
    int bn, ifm, ofm, ihw, id;

public:
    BlockDataLayer(int _bn, int _ifm, int _id, int _ihw): bn(_bn), ifm(_ifm), 
                                                          id(_id), ihw(_ihw) 
    {   
        assert(ihw > 0);
        assert(ifm > 0);
        assert( bn > 0);
        assert( id > 0);

        ofm = znn::phi::roundToSimd(ifm); 
    }

    void forward(float const* __restrict i, float* __restrict o, 
                 float const* __restrict dummy1, float const* __restrict dummy2)
    {
        typedef float (*out_tp)[ofm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
        typedef float const (*in_tp)[ifm][id][ihw][ihw];
 
        out_tp o_array = reinterpret_cast<out_tp>(o);
        in_tp i_array = reinterpret_cast<in_tp>(i);
         
//TODO: pragma parallel
        for (int b = 0; b < bn; ++b) {
            for (size_t d = 0; d < id; d++) {
                for (size_t c = 0; c < ofm; c++) {
                    auto s = c % SIMD_WIDTH;
                    auto f = c / SIMD_WIDTH;

                    for (int h = 0; h < ihw; h++) {
                        for (int w = 0; w < ihw; w++) {
                             /*int ti_1d = (&(o_array[b][f][d][h][w][s]) -
                                                  &(o_array[0][0][0][0][0][0]));
                             int bi_1d = (&(i_array[b][c][d][h][w]) -
                                                  &(i_array[0][0][0][0][0]));
                             std::cout << ti_1d << "--->" << bi_1d << std::endl;*/
                             if (c < ifm) { 
                                o_array[b][f][d][h][w][s] = i_array[b][c][d][h][w];
                             }
                             else {
                                o_array[b][f][d][h][w][s] = 0; 
                             }
                        }
                    }
                }
            }
        }
    }
};

}
}
