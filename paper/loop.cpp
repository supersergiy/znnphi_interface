template< int RBD, int RBH, int RBW >
void forward( float const * __restrict i,
              float const * __restrict w,
              float       * __restrict o,
              float bias ) const
{
    SIMD_FLOAT vout[RBD][RBH][RBW], vwt;

    #pragma unroll(RBD)
    for ( int rbd = 0; rbd < RBD; ++rbd )
    #pragma unroll(RBH)
    for ( int rbh = 0; rbh < RBH; ++rbh )
    #pragma unroll(RBW)
    for ( int rbw = 0; rbw < RBW; ++rbw )
    {
        vout[rbd][rbh][rbw]
            = conditional_load<FIRST>
            ( bias,
              o + (rbw * W::os +
                   rbd * D::os +
                   rbh * H::os) * SIMD_WIDTH );
    }

    for ( int kd = 0; kd < CD::s; ++kd )
    for ( int kh = 0; kh < CH::s; ++kh )
    for ( int s  = 0;  s < SW   ; ++s  )
    for ( int kw = 0; kw < CW::s; ++kw )
    {
        vwt = SIMD_LOAD( w +
                         ((kh * CW::s + kw
                           + kd * CW::s * CH::s)
                          * SIMD_WIDTH + s)
                         * SIMD_WIDTH );

        #pragma unroll(RBD)
        for ( int rbd = 0; rbd < RBD; ++rbd )
        #pragma unroll(RBH)
        for ( int rbh = 0; rbh < RBH; ++rbh )
        #pragma unroll(RBW)
        for ( int rbw = 0; rbw < RBW; ++rbw )
        {
            vout[rbd][rbh][rbw] = SIMD_FMADD
                ( vwt,
                  SIMD_SET1(i[( (kd + rbd) * D::is +
                                (kh + rbh) * H::is +
                                (kw + rbw) * W::is) *
                              SIMD_WIDTH + s]),
                  vout[rbd][rbh][rbw]);
        }
    }

    #pragma unroll(RBD)
    for ( int rbd = 0; rbd < RBD; ++rbd )
    #pragma unroll(RBH)
    for ( int rbh = 0; rbh < RBH; ++rbh )
    #pragma unroll(RBW)
    for ( int rbw = 0; rbw < RBW; ++rbw )
    {
        SIMD_STORE( o + (rbw * W::os + rbd * D::os + rbh * H::os)
                    * SIMD_WIDTH,
                    vout[rbd][rbh][rbw] );
    }
}
