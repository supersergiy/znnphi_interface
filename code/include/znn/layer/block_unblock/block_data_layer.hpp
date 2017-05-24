
namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct block_data_layer{
private:
    int bn, fm, ihw, id;

    block_data(int _bn, int _fm, int _ihw, int _id): bn(_bn), fm(_fm), id(_id), ihw(_ihw) 
    {   
        assert(fm % SIMD_WIDTH == 0); 
        assert( bn > 0);
        assert( fm > 0);
        assert( id > 0);
        assert(ihw > 0);
    }

public:
    void execute(float const* __restrict i, float* __restirct o)
    {
        typedef float (*out_tp)[fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
        typedef float const (*in_tp)[fm][id][ihw][ihw];
 
        out_tp o_array = reinterpret_cast<top_tp>(o);
        in_tp i_array = reinterpret_cast<bot_tp const>(i);

//TODO: pragma parallel
        for (int b = 0; b < bn; ++b) {
            for (size_t d = 0; d < id; d++) {
                for (size_t c = 0; c < fm; c++) {
                    auto s = c % SIMD_WIDTH;
                    auto f = c / SIMD_WIDTH;

                    for (int h = 0; h < ihw; h++) {
                        for (int w = 0; w < ihw; w++) {
                            /*int ti_1d = (&(top_array[n][f][d][h][w][s]) -
                                                  &(top_array[0][0][0][0][0][0]));
                             int bi_1d = (&(bottom_array[n][c][d][h][w]) -
                                                  &(bottom_array[0][0][0][0][0]));
                             std::cout << ti_1d << "--->" << bi_1d << std::endl;*/
                             o_array[n][f][d][h][w][s] = i_array[n][c][d][h][w];
                        }
                    }
                }
            }
        }
    }
};

}
}
