#include "../conv_wrapper.hpp"
#include <iostream>
#include <omp.h>
#include <vector>

int main()
{
	znn::phi::ConvWrapper znnphi_convs;
	std::cout << "let's go" << std::endl;
    znnphi_convs.forward(NULL, NULL, NULL, NULL);
}

