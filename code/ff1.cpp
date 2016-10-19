#include "znn/types.hpp"
#include "znn/layer/dimension.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/layer/task.hpp"
#include "znn/layer/conv/fwd2/blocking.hpp"
#include "znn/util/conditional_load.hpp"

namespace znn { namespace phi {

// kernel shape is
// K[D][H][W][IN_FMAP][OUT_FMAP] with IN=OUT=16

// input and output shapes are
// Im[D][H][W][FMAP]

// for all valid d, h, w, kd, kh, kw, if, of
// Out[d][h][w][of] += K[kd][kh][kw][if][of] * In[d+kd][h+kh][w+kw][if];

template< bool FIRST,                      // load or set to zero
          long_t SW,                       // number of input featuremaps avx
          class D , class H , class W ,    // out size and in/out strides
          class CD, class CH, class CW     // convolution params
          >
class fwd_work
{
private:
    //using RB = fwd_blocking_t<D::s,H::s,W::s>;
    struct RB
    {
        static const long_t w = 29;
        static const long_t h = 1;
        static const long_t d = 1;
    };

    template< long_t D, long_t H, long_t W >
    struct signuper
    {
        signuper()
        {
            std::cout << "SIGNED: " << D << ' ' << H << ' ' << W << '\n';
        }
    };


public:

    template< long_t RBD, long_t RBH, long_t RBW >
    typename std::enable_if< RBD==0 || RBH==0 || RBW==0,void >::type
    static chunk( float const * __restrict,
           float       * __restrict,
           float const * __restrict,
           float const * __restrict )
    {
    }

    template< long_t RBD, long_t RBH, long_t RBW >
    typename std::enable_if< (RBD>0) && (RBH>0) && (RBW>0),void >::type
    static chunk( float const * __restrict i,
           float       * __restrict o,
           float const * __restrict k,
           float const * __restrict b)

    {
        static signuper<RBD,RBH,RBW> s;

        SIMD_FLOAT vin[RBW+2];
        SIMD_FLOAT vout[RBW], vwt[3]; // in registers

#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            vout[rbw] = conditional_load_or_bias<FIRST>
                ( o + (rbw * W::os), b );
        }



        for ( long_t kd = 0; kd < CD::s; ++kd )

        for ( long_t kh = 0; kh < CH::s; ++kh )

        for ( long_t s  = 0;  s < SW; ++s  )
        {

#pragma unroll(CW::s)
            for ( long_t kw = 0; kw < CW::s; ++kw )
            {
                vwt[kw] = SIMD_LOAD( k +
                                     ((kh * CW::s + kw
                                       + kd * CW::s * CH::s) * SIMD_WIDTH + s)
                                     * SIMD_WIDTH );
            }

#pragma unroll(RBW+2)
            for ( long_t rbw = 0; rbw < RBW+2; ++rbw )
            {
                vin[rbw] = SIMD_SET1(i[( kd * D::is + kh * H::is + rbw * W::is) + s]);
            }

#pragma unroll(CW::s)
            for ( long_t kw = 0; kw < CW::s; ++kw )
#pragma unroll(RBW)
            for ( long_t rbw = 0; rbw < RBW; ++rbw )
            {
                vout[rbw] = SIMD_FMADD(vwt[kw],vin[kw+rbw],vout[rbw]);
            }
        }



#pragma unroll(RBW)
        for ( long_t rbw = 0; rbw < RBW; ++rbw )
        {
            SIMD_STORE( o + (rbw * W::os),
                        vout[rbw] );
        }
    }


    static long_t flops()
    {
        return CW::s * CH::s * CD::s * SW * D::s * H::s * W::s * SIMD_WIDTH * 2;
    }

    template< long_t DD, long_t HH >
    static void loopw( float const * __restrict i,
                       float       * __restrict o,
                       float const * __restrict k,
                       float const * __restrict b)
    {
        static const long_t FULL = W::s / RB::w;
        static const long_t PART = W::s % RB::w;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template chunk<DD,HH,RB::w>(i,o,k,b);
            i += RB::w * W::is * CW::stride;
            o += RB::w * W::os;
        }

        if ( PART )
        {
            fwd_work::template chunk<DD,HH,PART>(i,o,k,b);
        }
    }

    template< long_t DD >
    static void looph( float const * __restrict i,
                       float       * __restrict o,
                       float const * __restrict k,
                       float const * __restrict b)
    {
        static const long_t FULL = H::s / RB::h;
        static const long_t PART = H::s % RB::h;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template loopw<DD,RB::h>(i,o,k,b);
            i += RB::h * H::is * CH::stride;
            o += RB::h * H::os;
        }

        if ( PART )
        {
            fwd_work::template loopw<DD,PART>(i,o,k,b);
        }
    }


    static void loopd( float const * __restrict i,
                       float       * __restrict o,
                       float const * __restrict k,
                       float const * __restrict b)
    {
        static const long_t FULL = D::s / RB::d;
        static const long_t PART = D::s % RB::d;

        for ( long_t d = 0; d < FULL; ++d )
        {
            fwd_work::template looph<RB::d>(i,o,k,b);
            i += RB::d * D::is * CD::stride;
            o += RB::d * D::os;
        }

        if ( PART )
        {
            fwd_work::template looph<PART>(i,o,k,b);
        }
    }

public:
    static void execute( float const * i,
                         float       * o,
                         float const * k,
                         float const * b)
    {
        loopd(i,o,k,b);
        //std::cout << "Work\n";
    }
};


template< bool FIRST,                      // load or set to zero
          long_t SW,                       // number of input featuremaps avx
          long_t IFM, long_t IFM_STRIDE,
          long_t OFM, long_t OFM_STRIDE,
          long_t KIN_STRIDE, long_t KOUT_STRIDE,
          class D , class H , class W ,    // out size and in/out strides
          class CD, class CH, class CW     // convolution params
          >
class fwd_net
{
private:
    using work_type = fwd_work<FIRST,SW,D,H,W,CD,CH,CW>;


    template< long_t NIFM, long_t NOFM >
    static typename std::enable_if< NIFM==0 || NOFM==0,void >::type
    recurse( float const *,
             float       *,
             float const *,
             float const *)
    {}

    template< long_t NIFM, long_t NOFM >
    static typename std::enable_if< NIFM==1 && NOFM==1,void >::type
    recurse( float const * i,
             float       * o,
             float const * k,
             float const * b)
    {
        work_type::execute(i,o,k,b);
    }

    template< long_t NIFM, long_t NOFM >
    static typename std::enable_if< NIFM>=1 && NOFM>=1 && (NIFM+NOFM>2),void >::type
    recurse( float const * i,
             float       * o,
             float const * k,
             float const * b)
    {
        static const long_t IFM_FIRST  = (NIFM/2);
        static const long_t IFM_SECOND = NIFM - IFM_FIRST;

        static const long_t OFM_FIRST  = (NOFM/2);
        static const long_t OFM_SECOND = NOFM - OFM_FIRST;

        fwd_net::template recurse<IFM_FIRST,OFM_FIRST>(
            i, o, k, b );

        fwd_net::template recurse<IFM_FIRST,OFM_SECOND>(
            i,
            o + OFM_FIRST * OFM_STRIDE,
            k + OFM_FIRST * KOUT_STRIDE,
            b + OFM_FIRST * SIMD_WIDTH );

        fwd_net::template recurse<IFM_SECOND,OFM_SECOND>(
            i + IFM_FIRST * IFM_STRIDE,
            o + OFM_FIRST * OFM_STRIDE,
            k + OFM_FIRST * KOUT_STRIDE + IFM_FIRST * KIN_STRIDE,
            b + OFM_FIRST * SIMD_WIDTH );

        fwd_net::template recurse<IFM_SECOND,OFM_FIRST>(
            i + IFM_FIRST * IFM_STRIDE,
            o,
            k + IFM_FIRST * KIN_STRIDE,
            b );
    }


public:
    static long_t flops()
    {
        return work_type::flops() * IFM * OFM;
    }

    static void execute( float const * i,
                         float       * o,
                         float const * k,
                         float const * b)
    {
        //loopd(i,o,k,b);
        fwd_net::template recurse<IFM,OFM>(i,o,k,b);
        //std::cout << "Work\n";
    }
};


template< bool FIRST,                      // load or set to zero
          class D , class H , class W ,    // out size and in/out strides
          class CD, class CH, class CW     // convolution params
          >
class fwd_work<FIRST, 0, D, H, W, CD, CH, CW>
{

public:
    static long_t flops()
    {
        return 0;
    }

    static void execute( float const *,
                         float       *,
                         float const *,
                         float const * )
    {}
};


template< long_t OD, long_t OHW,
          long_t KD, long_t KHW >
double benchmark_single_forward()
{
    static const long_t OH = OHW;
    static const long_t OW = OHW;

    static const long_t ID = OD + KD - 1;
    static const long_t IH = OH + KHW - 1;
    static const long_t IW = OW + KHW - 1;

    static const long_t IFM = 16;
    static const long_t OFM = 16;

    using work_type = fwd_net< false,
                               SIMD_WIDTH,
                               IFM, ID*IH*IW*SIMD_WIDTH,
                               OFM, OD*OH*OW*SIMD_WIDTH,
                               SIMD_WIDTH*SIMD_WIDTH*KD*KHW*KHW,
                               SIMD_WIDTH*SIMD_WIDTH*KD*KHW*KHW*IFM,

        dimension<OD,IH*IW*SIMD_WIDTH,OW*OH*SIMD_WIDTH>,
        dimension<OH,IW*SIMD_WIDTH,OW*SIMD_WIDTH>,
        dimension<OW,SIMD_WIDTH,SIMD_WIDTH>,
        conv_traits<KD,1,1>,
        conv_traits<KHW,1,1>,
        conv_traits<KHW,1,1>>;


    hbw_array<float> in  (one_init, SIMD_WIDTH*ID*IH*IW*IFM);
    hbw_array<float> ker (one_init, SIMD_WIDTH*SIMD_WIDTH*KD*KHW*KHW*IFM*OFM);
    hbw_array<float> out (one_init, SIMD_WIDTH*OD*OH*OW*OFM);
    hbw_array<float> bias(one_init, SIMD_WIDTH);

    for ( long_t i = 0; i < 6; ++i )
    {
        work_type::execute(in.data(),out.data(),ker.data(),bias.data());
    }

    long_t iters = 50;
    auto begin = std::chrono::high_resolution_clock::now();

    for ( long_t i = 0; i < iters; ++i )
    {
        //FLOPSS = 0;
        work_type::execute(in.data(),out.data(),ker.data(),bias.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
        (end-begin).count();

    double secs  = static_cast<double>(duration) / 1000000;

    double gflops = work_type::flops();
    gflops /= 1000000000;

    std::cout << "Secs   : " << (secs/iters) << "\n";
    std::cout << "GFLOP/s: " << (gflops*iters/secs) << "\n\n";

    //std::cout << FLOPSS << " : " << work_type::flops() << "\n";
    //FLOPSS = 0;

    return secs/iters;
}


}} // namespace znn:phi

using namespace znn::phi;

int main()
{
    //benchmark_single_forward<1,1008,1,3>();
    //benchmark_single_forward<1,108,1,3>();
    //benchmark_single_forward<1,1204,1,3>();
    //benchmark_single_forward<1,210,1,3>();
    //benchmark_single_forward<1,53,1,3>();

    benchmark_single_forward<1,30,1,3>();
    //benchmark_single_forward<1,14,1,3>();

    // benchmark_single_forward<1,58,1,2>();
    // benchmark_single_forward<1,15,1,2>();


}
