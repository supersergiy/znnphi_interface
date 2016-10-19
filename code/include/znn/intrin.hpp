#pragma once

#include "znn/types.hpp"
#include <x86intrin.h>

#if defined(ZNN_AVX512)

#define SIMD_WIDTH 16

#define _FMADD  _mm512_fmadd_ps
#define _FMSUB  _mm512_fmsub_ps
#define _FNMADD _mm512_fnmadd_ps
#define _FNMSUB _mm512_fnmsub_ps

#define _SET1   _mm512_set1_ps
#define _MUL    _mm512_mul_ps
#define _ADD    _mm512_add_ps
#define _SUB    _mm512_sub_ps

#define _LOAD   _mm512_load_ps

#define SIMD_FMADD  _mm512_fmadd_ps
#define SIMD_FMSUB  _mm512_fmsub_ps
#define SIMD_FNMADD _mm512_fnmadd_ps
#define SIMD_FNMSUB _mm512_fnmsub_ps

#define SIMD_SET1   _mm512_set1_ps
#define SIMD_MUL    _mm512_mul_ps
#define SIMD_ADD    _mm512_add_ps
#define SIMD_SUB    _mm512_sub_ps

#define SIMD_LOAD   _mm512_load_ps
#define SIMD_STORE  _mm512_store_ps
#define SIMD_STREAM _mm512_stream_ps
#define SIMD_ZERO   _mm512_setzero_ps

#define SIMD_MAX    _mm512_max_ps

#define SIMD_FLOAT  __m512

#define SIMD_MAX_BLOCK 12
#define SIMD_W_BLOCK   12

#define SIMD_NUM_REGISTERS 32

#define SIMD_CMP(a,b) _mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ)
#define SIMD_MASK_ADD(r,m,a,b) _mm512_mask_add_ps(r,m,a,b)

#elif defined(ZNN_KNC)

#define SIMD_WIDTH 16

#define _FMADD  _mm512_fmadd_ps
#define _FMSUB  _mm512_fmsub_ps
#define _FNMADD _mm512_fnmadd_ps
#define _FNMSUB _mm512_fnmsub_ps

#define _SET1   _mm512_set1_ps
#define _MUL    _mm512_mul_ps
#define _ADD    _mm512_add_ps
#define _SUB    _mm512_sub_ps

#define _LOAD   _mm512_load_ps

#define SIMD_FMADD  _mm512_fmadd_ps
#define SIMD_FMSUB  _mm512_fmsub_ps
#define SIMD_FNMADD _mm512_fnmadd_ps
#define SIMD_FNMSUB _mm512_fnmsub_ps

#define SIMD_SET1   _mm512_set1_ps
#define SIMD_MUL    _mm512_mul_ps
#define SIMD_ADD    _mm512_add_ps
#define SIMD_SUB    _mm512_sub_ps

#define SIMD_LOAD   _mm512_load_ps
#define SIMD_STORE  _mm512_store_ps
#define SIMD_STREAM _mm512_stream_ps
#define SIMD_ZERO   _mm512_setzero_ps

#define SIMD_MAX    _mm512_max_ps

#define SIMD_FLOAT  __m512

#define SIMD_MAX_BLOCK 30
#define SIMD_W_BLOCK   14

#define SIMD_NUM_REGISTERS 32

#define SIMD_CMP(a,b) _mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ)
#define SIMD_MASK_ADD(r,m,a,b) _mm512_mask_add_ps(r,m,a,b)

#elif defined(ZNN_AVX2)

#define SIMD_WIDTH 8

#define _FMADD  _mm256_fmadd_ps
#define _FMSUB  _mm256_fmsub_ps
#define _FNMADD _mm256_fnmadd_ps
#define _FNMSUB _mm256_fnmsub_ps

#define _SET1   _mm256_set1_ps
#define _MUL    _mm256_mul_ps
#define _ADD    _mm256_add_ps
#define _SUB    _mm256_sub_ps

#define _LOAD   _mm256_load_ps

#define SIMD_FMADD  _mm256_fmadd_ps
#define SIMD_FMSUB  _mm256_fmsub_ps
#define SIMD_FNMADD _mm256_fnmadd_ps
#define SIMD_FNMSUB _mm256_fnmsub_ps

#define SIMD_SET1   _mm256_set1_ps
#define SIMD_MUL    _mm256_mul_ps
#define SIMD_ADD    _mm256_add_ps
#define SIMD_SUB    _mm256_sub_ps

#define SIMD_LOAD   _mm256_load_ps
#define SIMD_STORE  _mm256_store_ps
#define SIMD_STREAM _mm256_stream_ps
#define SIMD_ZERO   _mm256_setzero_ps

#define SIMD_MAX    _mm256_max_ps

#define SIMD_FLOAT  __m256

#define SIMD_MAX_BLOCK 7
#define SIMD_W_BLOCK   7

#define SIMD_NUM_REGISTERS 16

#elif defined(ZNN_AVX)

#define SIMD_WIDTH 8

#define _FMADD(a,b,c)  _mm256_add_ps(_mm256_mul_ps(a,b),c)
#define _FMSUB(a,b,c)  _mm256_sub_ps(_mm256_mul_ps(a,b),c)
#define _FNMADD(a,b,c) _mm256_sub_ps(c,_mm256_mul_ps(a,b))
#define _FNMSUB(a,b,c) _mm256_sub_ps(_mm256_setzero_ps(),_FMADD(a,b,c))

#define _SET1   _mm256_set1_ps
#define _MUL    _mm256_mul_ps
#define _ADD    _mm256_add_ps
#define _SUB    _mm256_sub_ps

#define _LOAD   _mm256_load_ps

#define SIMD_FMADD(a,b,c)  _mm256_add_ps(_mm256_mul_ps(a,b),c)
#define SIMD_FMSUB(a,b,c)  _mm256_sub_ps(_mm256_mul_ps(a,b),c)
#define SIMD_FNMADD(a,b,c) _mm256_sub_ps(c,_mm256_mul_ps(a,b))
#define SIMD_FNMSUB(a,b,c) _mm256_sub_ps(_mm256_setzero_ps(),SIMD_FMADD(a,b,c))

#define SIMD_SET1   _mm256_set1_ps
#define SIMD_MUL    _mm256_mul_ps
#define SIMD_ADD    _mm256_add_ps
#define SIMD_SUB    _mm256_sub_ps

#define SIMD_LOAD   _mm256_load_ps
#define SIMD_STORE  _mm256_store_ps
#define SIMD_STREAM _mm256_stream_ps
#define SIMD_ZERO   _mm256_setzero_ps

#define SIMD_MAX    _mm256_max_ps

#define SIMD_FLOAT  __m256

#define SIMD_MAX_BLOCK 7
#define SIMD_W_BLOCK   7

#define SIMD_NUM_REGISTERS 16

#else

#error "NEED SOME AVX DEFINED"

#endif

#define SIMD_PREFETCH_L1(address) _mm_prefetch(address,_MM_HINT_T0)

#define SIMD_PREFETCH_L2(address) _mm_prefetch(address,_MM_HINT_T1)
