#include "znet.hpp"
#include <iostream>

int main(void)
{
    znn::phi::Znet zn("./out/weights/");

    for (int i = 0; i < zn.input_size; i++) {
       zn.tensors["user_input"]->data()[i] = 0.0;
    }

    zn.forward();
    size_t out_size = zn.out_shape[0] * zn.out_shape[1] * zn.out_shape[2] *
                         zn.out_shape[3] * zn.out_shape[4];
    for (int i = 0; i < out_size; i++) {
       std::cout << zn.tensors["user_output"]->data()[i] << " "; 
       if (i  % 192 == 0) {
         std::cout << std::endl;
       }
    }
    return 0;
}
