#include <iostream>
#include <utility>
#include <tuple>
#include "znn/meta.hpp"
#include "znn/types.hpp"

namespace znn { namespace phi {

inline constexpr long_t smallest_prime_factor( long_t a )
{
    return ( a % 2 == 0 ) ? 2 : (
        ( a % 3 == 0 ) ? 3 : (
            ( a % 5 == 0 ) ? 5 : (
                ( a % 7 == 0 ) ? 7 : (
                    ( a % 11 == 0 ) ? 11 : (
                        ( a % 13 == 0 ) ? 13 : a )))));
}

template< class Problem >
struct upd_io_split_t
{
private:
    static constexpr long_t prime = smallest_prime_factor(Problem::threads);

public:

    static_assert( prime >= 1, "");

    using type = typename conditional_t<
        condition_t< prime == 1 >,
        type_wrapper_t< Problem >,
        condition_t< (Problem::i > Problem::o) && (Problem::i % prime == 0) >,
        upd_io_split_t< upd_io_problem_t< Problem::threads / prime,
                                          Problem::i / prime,
                                          Problem::o > >,
        condition_t< Problem::o % prime == 0 >,
        upd_io_split_t< upd_io_problem_t< Problem::threads / prime,
                                          Problem::i,
                                          Problem::o / prime > >,
        condition_t< Problem::i % prime == 0 >,
        upd_io_split_t< upd_io_problem_t< Problem::threads / prime,
                                          Problem::i / prime,
                                          Problem::o > >,
        type_wrapper_t< Problem > >::type::type;
};



template< class >
struct upd_split_problem_t;


template< class, class >
struct upd_extractor_t;

template< class C, long_t... I >
struct upd_extractor_t< C, std::integer_sequence<long_t, I...>>
{
    using type = typename upd_problem_cat_t< typename upd_split_problem_t<
        typename C::template get<I>::type>::type... >::type;
};

template< class C >
struct upd_recursive_split_t
{
    using type = typename
        upd_extractor_t<C, std::make_integer_sequence<long_t, C::parts>>::type;
};

template< class Problem, long_t N >
struct upd_batch_split_t
{
private:

    static constexpr long_t blen = Problem::size::batch / N;
    static constexpr long_t badd = Problem::size::batch % N;

    static constexpr long_t bioff =
        Problem::ioffset + Problem::ishape::batch * (blen+1) * badd;

    static constexpr long_t booff =
        Problem::ooffset + Problem::oshape::batch * (blen+1) * badd;


public:

    static constexpr long_t parts = N;

    template< long_t K >
    struct get
    {
        using type = typename std::conditional<(K<badd),
            upd_problem_t< Problem::threads / N,
                           upd_problem_size_t< blen+1,
                                               Problem::size::depth,
                                               Problem::size::height >,
                           typename Problem::ishape,
                           typename Problem::oshape,
                           Problem::ioffset + Problem::ishape::batch * (blen+1) * K,
                           Problem::ooffset + Problem::oshape::batch * (blen+1) * K >,
            upd_problem_t< Problem::threads / N,
                           upd_problem_size_t< blen,
                                               Problem::size::depth,
                                               Problem::size::height >,
                           typename Problem::ishape,
                           typename Problem::oshape,
                           bioff + Problem::ishape::batch * blen * (K-badd),
                           booff + Problem::oshape::batch * blen * (K-badd) > >::type;
    };
};


template< class Problem, long_t N >
struct upd_depth_split_t
{
private:
    static constexpr long_t dlen = Problem::size::depth / N;
    static constexpr long_t dadd = Problem::size::depth % N;

    static constexpr long_t dioff =
        Problem::ioffset + Problem::ishape::depth * (dlen+1) * dadd;

    static constexpr long_t dooff =
        Problem::ooffset + Problem::oshape::depth * (dlen+1) * dadd;

public:

    static constexpr long_t parts = N;

    template< long_t K >
    struct get
    {
        using type = typename std::conditional<(K<dadd),
            upd_problem_t< Problem::threads / N,
                           upd_problem_size_t< Problem::size::batch,
                                               dlen+1,
                                               Problem::size::height >,
                           typename Problem::ishape,
                           typename Problem::oshape,
                           Problem::ioffset + Problem::ishape::depth * (dlen+1) * K,
                           Problem::ooffset + Problem::oshape::depth * (dlen+1) * K >,
            upd_problem_t< Problem::threads / N,
                           upd_problem_size_t< Problem::size::batch,
                                               dlen,
                                               Problem::size::height >,
                           typename Problem::ishape,
                           typename Problem::oshape,
                           dioff + Problem::ishape::depth * dlen * (K-dadd),
                           dooff + Problem::oshape::depth * dlen * (K-dadd) > >::type;
    };
};

template< class Problem, long_t N >
struct upd_height_split_t
{
private:
    static constexpr long_t hlen = Problem::size::height / N;
    static constexpr long_t hadd = Problem::size::height % N;

    static constexpr long_t hioff =
        Problem::ioffset + Problem::ishape::height * (hlen+1) * hadd;

    static constexpr long_t hooff =
        Problem::ooffset + Problem::oshape::height * (hlen+1) * hadd;


public:

    static constexpr long_t parts = N;

    template< long_t K >
    struct get
    {
        using type = typename std::conditional<(K<hadd),
            upd_problem_t< Problem::threads / N,
                           upd_problem_size_t< Problem::size::batch,
                                               Problem::size::depth,
                                               hlen+1 >,
                           typename Problem::ishape,
                           typename Problem::oshape,
                           Problem::ioffset + Problem::ishape::height * (hlen+1) * K,
                           Problem::ooffset + Problem::oshape::height * (hlen+1) * K >,
            upd_problem_t< Problem::threads / N,
                           upd_problem_size_t< Problem::size::batch,
                                               Problem::size::depth,
                                               hlen >,
                           typename Problem::ishape,
                           typename Problem::oshape,
                           hioff + Problem::ishape::height * hlen * (K-hadd),
                           hooff + Problem::oshape::height * hlen * (K-hadd) > >::type;
    };
};


template<>
struct upd_split_problem_t<null_upd_problem_t>
{
    using type = upd_problems_t< null_upd_problem_t >;
};


template< class Problem >
struct upd_split_problem_t
{
private:
    static constexpr long_t prime = smallest_prime_factor(Problem::threads);

public:

    static_assert( prime >= 1, "" );

    using type = typename conditional_t<
        condition_t< prime == 1 >,
        type_wrapper_t< upd_problems_t< Problem > >,
        condition_t< Problem::size::batch % prime == 0 >,
        upd_recursive_split_t< upd_batch_split_t< Problem, prime >>,
        condition_t< Problem::size::depth % prime == 0 >,
        upd_recursive_split_t< upd_depth_split_t< Problem, prime >>,
        condition_t< Problem::size::height % prime == 0 >,
        upd_recursive_split_t< upd_height_split_t< Problem, prime >>,
        condition_t< (Problem::size::batch >= prime) >,
        upd_recursive_split_t< upd_batch_split_t< Problem, prime >>,
        condition_t< (Problem::size::depth >= prime) >,
        upd_recursive_split_t< upd_depth_split_t< Problem, prime >>,
        condition_t< (Problem::size::height >= prime) >,
        upd_recursive_split_t< upd_height_split_t< Problem, prime >>,
        type_wrapper_t< upd_problems_t< Problem > > >::type::type;
};





}} // namespace znn:phi


using namespace znn::phi;

int main()
{
    using xt = typename upd_io_split_t< upd_io_problem_t< 72, 32, 16 > >::type;

    std::cout << xt::threads << ' ' << xt::i << ' ' << xt::o << "\n";
    //std::cout << smallest_prime_factor(72) << "\n";

    std::cout << sizeof(upd_problems_t<>) << "\n";

    using myprob = upd_problem_t<
        9,
        upd_problem_size_t< 5, 5, 12 >,
        upd_ioshape_t< 60, 12, 1 >,
        upd_ioshape_t< 60, 12, 1 > >;

    //using prob = upd_problems_t<null_upd_problem_t,null_upd_problem_t,myprob>;

    //using prob2 = upd_problem_cat_t<prob,prob,prob>::type;

    // using prob2 = upd_extractor_t< upd_batch_split_t< myprob, 7 >,
    //                                std::make_integer_sequence<long_t, 7> >::type;

    using probs = upd_split_problem_t< myprob >::type;

    upd_problems_printer<probs>::print();
}
