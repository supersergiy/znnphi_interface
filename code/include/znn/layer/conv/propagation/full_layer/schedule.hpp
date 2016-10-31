#pragma once

#include "znn/layer/conv/propagation/executable.hpp"
#include "znn/layer/conv/propagation/sub_layer.hpp"
#include "znn/layer/conv/propagation/full_layer/problem.hpp"
#include "znn/layer/conv/propagation/full_layer/split.hpp"

#include <iostream>
#include <type_traits>

namespace znn
{
namespace phi
{
namespace propagation
{

template <class P>
struct serial_scheduler
{
private:
    using sub = typename P::sub_problem;
    static_assert(P::threads == 1, "number of threads not 1");

public:
    static void schedule(long_t t, exec_vector& ev)
    {
        // std::cout << "Scheduling on Thread " << t << " batch: " << sub::b_from
        //           << "-" << sub::b_from + sub::b_len - 1
        //           << " ofm: " << sub::ofm_from << "-"
        //           << sub::ofm_from + sub::ofm_len - 1 << " d: " << sub::d_from
        //           << "-" << sub::d_from + sub::d_len - 1
        //           << " h: " << sub::h_from << "-"
        //           << sub::h_from + sub::h_len - 1 << " w: " << sub::w_from
        //           << "-" << sub::w_from + sub::w_len - 1 << "\n";
        ev[t].push_back(&sub_layer<P>::execute);
    }
};

template <class P>
struct scheduler;

template <class P, long_t N>
struct split_scheduler
{
private:
    using splitted = split_t<P, N>;

    template <long_t                  K>
    static std::enable_if_t<(K == N)> schedule_part(long_t       t0, long_t,
                                                    exec_vector& ev)
    {
        scheduler<typename splitted::rest_t>::schedule(t0, ev);
    }

    template <long_t                 K>
    static std::enable_if_t<(K < N)> schedule_part(long_t t0, long_t t,
                                                   exec_vector& ev)
    {
        scheduler<typename splitted::template part_t<K>>::schedule(t, ev);
        split_scheduler::template schedule_part<K + 1>(t0, t + P::threads / N,
                                                       ev);
    }

public:
    static void schedule(long_t t, exec_vector& ev)
    {
        split_scheduler::template schedule_part<0>(t, t, ev);
    }
};

template <class P>
struct scheduler
{
private:
    using type = std::conditional_t<
        P::threads == 1, serial_scheduler<P>,
        std::conditional_t<
            (P::threads % 2) == 0, split_scheduler<P, 2>,
            std::conditional_t<
                (P::threads % 3) == 0, split_scheduler<P, 3>,
                std::conditional_t<
                    (P::threads % 5) == 0, split_scheduler<P, 5>,
                    std::conditional_t<
                        (P::threads % 7) == 0, split_scheduler<P, 7>,
                        std::conditional_t<(P::threads % 11) == 0,
                                           split_scheduler<P, 11>,
                                           split_scheduler<P, P::threads>>>>>>>;

public:
    static void schedule(long_t t, exec_vector& ev) { type::schedule(t, ev); }
};

template <>
struct scheduler<null_problem_t>
{
    static void schedule(long_t, exec_vector&) {}
};

} // namespace propagation
} // namespace phi
} // namespace znn
