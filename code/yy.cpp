#include "znn/layer/conv/upd3/ioproblem.hpp"
#include "znn/layer/conv/upd3/problem.hpp"
#include "znn/layer/conv/upd3/plan.hpp"


using namespace znn::phi;

int main()
{
    // using prob1 = upd_io_problem_t< 144, 32, 32, upd_io_problem_strides<1,1,1,32> >;
    // //using prob2 = upd_io_problem_t< 3, 2, 2, int >;

    // using xx = upd_io_split_problem_t<prob1>;

    // upd_io_problems_printer<xx>::print();

    // using onep = typename std::tuple_element<0,xx>::type;

    // using myprob = upd_problem_t<
    //     onep::threads,
    //     upd_problem_size_t< 1, 2, 4 >,
    //     upd_ioshape_t< 60, 12, 1 >,
    //     upd_ioshape_t< 60, 12, 1 > >;


    // using yy = upd_split_problem_t<myprob>;



    // upd_problems_printer<yy>::print();

    upd_plan< 144, 32, 128, 64,
              18, 58, 58,
              16, 56, 56,
              3, 3, 3 > plan;

}
