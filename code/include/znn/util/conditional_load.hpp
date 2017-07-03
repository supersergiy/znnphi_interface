#pragma once

#include "znn/intrin.hpp"
#include "znn/types.hpp"

#include <type_traits>

namespace znn
{
namespace phi
{

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<ZERO, SIMD_FLOAT>::type
    conditional_load(float const*)
{
    return SIMD_ZERO();
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<!ZERO, SIMD_FLOAT>::type
    conditional_load(float const* address)
{
    return SIMD_LOAD(address);
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<ZERO, SIMD_FLOAT>::type
    conditional_load_or_bias(float const*, float const* b)
{
#ifdef ADD_CONV_RESULT_TO_OUTPUT 
    return SIMD_LOAD(b);
#else
    
#endif
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<!ZERO, SIMD_FLOAT>::type
    conditional_load_or_bias(float const* address, float const*)
{
    return SIMD_LOAD(address);
}

template <bool FIRST>
__attribute__((always_inline)) inline
    typename std::enable_if<!FIRST, SIMD_FLOAT>::type
    load_or_set_initial_value(float const* address, float const*, float const* s) 
{
    return SIMD_LOAD(address);
}

template <bool FIRST>
__attribute__((always_inline)) inline
    typename std::enable_if<FIRST, SIMD_FLOAT>::type
    load_or_set_initial_value(float const* address, float const*b, float const* s)
{
   return SIMD_LOAD(b);
                       
   //return (s == NULL? SIMD_LOAD(b) :
                       //SIMD_FMADD(SIMD_LOAD(address), SIMD_LOAD(s), SIMD_LOAD(b)));
}


}
} // namespace znn:phi
