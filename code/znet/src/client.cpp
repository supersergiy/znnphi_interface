#include "znet.hpp"

int main(void)
{
    znn::phi::Znet zn("./out/weights/");
    zn.forward();
    return 0;
}
