#include <assert.h>
#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>


namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct UnblockDataLayer: public Layer{
private:
    int bn, ifm, ofm, ihw, id;

public:
    UnblockDataLayer(int _bn, int _ofm, int _id, int _ihw): bn(_bn), 
                                                           ofm(_ofm), id(_id), ihw(_ihw) 
    {   
        assert(ihw > 0);
        assert(ofm > 0);
        assert( bn > 0);
        assert( id > 0);

        ifm = znn::phi::roundToSimd(ofm);
    }

    void forward(float const* __restrict i, float* __restrict o, 
                 float const* __restrict dummy1, float const* __restrict dummy2)
    {
        typedef float const (*in_tp)[ifm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
        typedef float (*out_tp)[ofm][id][ihw][ihw];
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
                            /*int ti_1d = (&(i_array[b][f][d][h][w][s]) -
                                                  &(i_array[0][0][0][0][0][0]));
                             int bi_1d = (&(o_array[b][c][d][h][w]) -
                                                  &(o_array[0][0][0][0][0]));
                             std::cout << ti_1d << "--->" << bi_1d << std::endl;*/
                             o_array[b][c][d][h][w] = i_array[b][f][d][h][w][s];
                        }
                    }
                }
            }
        }
    }
};

}
}
