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
    return SIMD_LOAD(b);
}

template <bool ZERO>
__attribute__((always_inline)) inline
    typename std::enable_if<!ZERO, SIMD_FLOAT>::type
    conditional_load_or_bias(float const* address, float const*)
{
    return SIMD_LOAD(address);
}

template <bool FIRST, bool ADD_OR_OVERWRITE>
__attribute__((always_inline)) inline
    typename std::enable_if<!FIRST, SIMD_FLOAT>::type
    load_or_set_initial_value(float const* address, float const*, float const* s) 
{
    return SIMD_LOAD(address);
}

template <bool FIRST, bool ADD_OR_OVERWRITE>
__attribute__((always_inline)) inline
    typename std::enable_if<FIRST && ADD_OR_OVERWRITE, SIMD_FLOAT>::type
    load_or_set_initial_value(float const* address, float const*b, float const* s)
{
   return SIMD_FMADD(SIMD_LOAD(address), SIMD_LOAD(s), SIMD_LOAD(b));
}

template <bool FIRST, bool ADD_OR_OVERWRITE>
__attribute__((always_inline)) inline
    typename std::enable_if<FIRST && !ADD_OR_OVERWRITE, SIMD_FLOAT>::type
    load_or_set_initial_value(float const* address, float const*b, float const* s)
{
   return SIMD_LOAD(b);
}


}
} // namespace znn:phi
