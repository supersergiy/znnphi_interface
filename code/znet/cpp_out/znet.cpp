#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include "znet.hpp"


znn::phi::Znet::Znet(void)
{
	layers["conv0_d1"] = new znn::phi::ConvWrapper(1, 32, 40, 18, 96, 1, 3, 0);
	layers["conv0_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	layers["conv0_d3"] = new znn::phi::ConvWrapper(1, 48, 64, 18, 24, 1, 3, 0);
	layers["conv0_d2"] = new znn::phi::ConvWrapper(1, 40, 48, 18, 48, 1, 3, 0);
	layers["conv0_d5"] = new znn::phi::ConvWrapper(1, 80, 96, 18, 6, 1, 3, 0);
	layers["conv0_d4"] = new znn::phi::ConvWrapper(1, 64, 80, 18, 12, 1, 3, 0);
	layers["conv4_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0);
	layers["conv4_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0);
	layers["conv4_d1"] = new znn::phi::ConvWrapper(1, 40, 40, 18, 96, 1, 3, 0);
	layers["conv5_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	layers["conv4_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	layers["conv6_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1);
	layers["conv6_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1);
	layers["convf6_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0);
	layers["convf6_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0);
	layers["conv6_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1);
	layers["convf5_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0);
	layers["convf5_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0);
	layers["convf5_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0);
	layers["conv2_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 3, 1, 1);
	layers["conv2_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1);
	layers["conv2_d1"] = new znn::phi::ConvWrapper(1, 40, 40, 18, 96, 1, 3, 0);
	layers["conv4_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0);
	layers["conv6_d1"] = new znn::phi::ConvWrapper(1, 40, 40, 18, 96, 1, 3, 0);
	layers["convf6_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0);
	layers["conv6_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	layers["conv5_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1);
	layers["conv5_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1);
	layers["conv5_d1"] = new znn::phi::ConvWrapper(1, 40, 40, 18, 96, 1, 3, 0);
	layers["conv5_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1);
	layers["conv1_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1);
	layers["conv1_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 3, 1, 1);
	layers["conv1_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1);
	layers["conv1_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1);
	layers["conv1_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	layers["conv1_d1"] = new znn::phi::ConvWrapper(1, 40, 40, 18, 96, 1, 3, 0);
	layers["convi"] = new znn::phi::ConvWrapper(1, 8, 32, 18, 192, 1, 5, 0);
	layers["conv7_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	layers["convf1_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0);
	layers["convf1_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0);
	layers["convf1_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 1, 3, 0);
	layers["convf1_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0);
	layers["convf2_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0);
	layers["convf2_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 1, 3, 0);
	layers["convf2_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0);
	layers["convf2_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0);
	layers["conv2_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1);
	layers["conv2_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1);
	layers["conv2_d0"] = new znn::phi::ConvWrapper(1, 32, 32, 18, 192, 1, 3, 0);
	
}


void znn::phi::Znet::forward(void)
{
	std::cout << "1\n";
	layers["convi"]->forward(tensors["input"]->data(), tensors["convi"]->data(), weights["convi_kernel"]->data(), weights["convi_bias"]->data());
	std::cout << "3\n";
	layers["conv0_d0"]->forward(tensors["convi"]->data(), tensors["conv0_d0"]->data(), weights["conv0_d0_kernel"]->data(), weights["conv0_d0_bias"]->data());
	std::cout << "7\n";
	layers["conv1_d0"]->forward(tensors["conv0_d0"]->data(), tensors["conv1_d0"]->data(), weights["conv1_d0_kernel"]->data(), weights["conv1_d0_bias"]->data());
	std::cout << "11\n";
	layers["conv2_d0"]->forward(tensors["conv1_d0"]->data(), tensors["conv2_d0"]->data(), weights["conv2_d0_kernel"]->data(), weights["conv2_d0_bias"]->data());
	std::cout << "17\n";
	layers["conv0_d1"]->forward(tensors["pool_d1"]->data(), tensors["conv0_d1"]->data(), weights["conv0_d1_kernel"]->data(), weights["conv0_d1_bias"]->data());
	std::cout << "21\n";
	layers["conv1_d1"]->forward(tensors["conv0_d1"]->data(), tensors["conv1_d1"]->data(), weights["conv1_d1_kernel"]->data(), weights["conv1_d1_bias"]->data());
	std::cout << "25\n";
	layers["conv2_d1"]->forward(tensors["conv1_d1"]->data(), tensors["conv2_d1"]->data(), weights["conv2_d1_kernel"]->data(), weights["conv2_d1_bias"]->data());
	std::cout << "31\n";
	layers["conv0_d2"]->forward(tensors["pool_d2"]->data(), tensors["conv0_d2"]->data(), weights["conv0_d2_kernel"]->data(), weights["conv0_d2_bias"]->data());
	std::cout << "35\n";
	layers["convf1_d2"]->forward(tensors["conv0_d2"]->data(), tensors["convf1_d2"]->data(), weights["convf1_d2_kernel"]->data(), weights["convf1_d2_bias"]->data());
	std::cout << "36\n";
	layers["conv1_d2"]->forward(tensors["convf1_d2"]->data(), tensors["conv1_d2"]->data(), weights["conv1_d2_kernel"]->data(), weights["conv1_d2_bias"]->data());
	std::cout << "40\n";
	layers["convf2_d2"]->forward(tensors["conv1_d2"]->data(), tensors["convf2_d2"]->data(), weights["convf2_d2_kernel"]->data(), weights["convf2_d2_bias"]->data());
	std::cout << "41\n";
	layers["conv2_d2"]->forward(tensors["convf2_d2"]->data(), tensors["conv2_d2"]->data(), weights["conv2_d2_kernel"]->data(), weights["conv2_d2_bias"]->data());
	std::cout << "47\n";
	layers["conv0_d3"]->forward(tensors["pool_d3"]->data(), tensors["conv0_d3"]->data(), weights["conv0_d3_kernel"]->data(), weights["conv0_d3_bias"]->data());
	std::cout << "51\n";
	layers["convf1_d3"]->forward(tensors["conv0_d3"]->data(), tensors["convf1_d3"]->data(), weights["convf1_d3_kernel"]->data(), weights["convf1_d3_bias"]->data());
	std::cout << "52\n";
	layers["conv1_d3"]->forward(tensors["convf1_d3"]->data(), tensors["conv1_d3"]->data(), weights["conv1_d3_kernel"]->data(), weights["conv1_d3_bias"]->data());
	std::cout << "56\n";
	layers["convf2_d3"]->forward(tensors["conv1_d3"]->data(), tensors["convf2_d3"]->data(), weights["convf2_d3_kernel"]->data(), weights["convf2_d3_bias"]->data());
	std::cout << "57\n";
	layers["conv2_d3"]->forward(tensors["convf2_d3"]->data(), tensors["conv2_d3"]->data(), weights["conv2_d3_kernel"]->data(), weights["conv2_d3_bias"]->data());
	std::cout << "63\n";
	layers["conv0_d4"]->forward(tensors["pool_d4"]->data(), tensors["conv0_d4"]->data(), weights["conv0_d4_kernel"]->data(), weights["conv0_d4_bias"]->data());
	std::cout << "67\n";
	layers["convf1_d4"]->forward(tensors["conv0_d4"]->data(), tensors["convf1_d4"]->data(), weights["convf1_d4_kernel"]->data(), weights["convf1_d4_bias"]->data());
	std::cout << "68\n";
	layers["conv1_d4"]->forward(tensors["convf1_d4"]->data(), tensors["conv1_d4"]->data(), weights["conv1_d4_kernel"]->data(), weights["conv1_d4_bias"]->data());
	std::cout << "72\n";
	layers["convf2_d4"]->forward(tensors["conv1_d4"]->data(), tensors["convf2_d4"]->data(), weights["convf2_d4_kernel"]->data(), weights["convf2_d4_bias"]->data());
	std::cout << "73\n";
	layers["conv2_d4"]->forward(tensors["convf2_d4"]->data(), tensors["conv2_d4"]->data(), weights["conv2_d4_kernel"]->data(), weights["conv2_d4_bias"]->data());
	std::cout << "79\n";
	layers["conv0_d5"]->forward(tensors["pool_d5"]->data(), tensors["conv0_d5"]->data(), weights["conv0_d5_kernel"]->data(), weights["conv0_d5_bias"]->data());
	std::cout << "83\n";
	layers["convf1_d5"]->forward(tensors["conv0_d5"]->data(), tensors["convf1_d5"]->data(), weights["convf1_d5_kernel"]->data(), weights["convf1_d5_bias"]->data());
	std::cout << "84\n";
	layers["conv1_d5"]->forward(tensors["convf1_d5"]->data(), tensors["conv1_d5"]->data(), weights["conv1_d5_kernel"]->data(), weights["conv1_d5_bias"]->data());
	std::cout << "88\n";
	layers["convf2_d5"]->forward(tensors["conv1_d5"]->data(), tensors["convf2_d5"]->data(), weights["convf2_d5_kernel"]->data(), weights["convf2_d5_bias"]->data());
	std::cout << "89\n";
	layers["conv2_d5"]->forward(tensors["convf2_d5"]->data(), tensors["conv2_d5"]->data(), weights["conv2_d5_kernel"]->data(), weights["conv2_d5_bias"]->data());
	std::cout << "99\n";
	layers["conv4_d4"]->forward(tensors["Eltwise1"]->data(), tensors["conv4_d4"]->data(), weights["conv4_d4_kernel"]->data(), weights["conv4_d4_bias"]->data());
	std::cout << "103\n";
	layers["convf5_d4"]->forward(tensors["conv4_d4"]->data(), tensors["convf5_d4"]->data(), weights["convf5_d4_kernel"]->data(), weights["convf5_d4_bias"]->data());
	std::cout << "104\n";
	layers["conv5_d4"]->forward(tensors["convf5_d4"]->data(), tensors["conv5_d4"]->data(), weights["conv5_d4_kernel"]->data(), weights["conv5_d4_bias"]->data());
	std::cout << "108\n";
	layers["convf6_d4"]->forward(tensors["conv5_d4"]->data(), tensors["convf6_d4"]->data(), weights["convf6_d4_kernel"]->data(), weights["convf6_d4_bias"]->data());
	std::cout << "109\n";
	layers["conv6_d4"]->forward(tensors["convf6_d4"]->data(), tensors["conv6_d4"]->data(), weights["conv6_d4_kernel"]->data(), weights["conv6_d4_bias"]->data());
	std::cout << "119\n";
	layers["conv4_d3"]->forward(tensors["Eltwise2"]->data(), tensors["conv4_d3"]->data(), weights["conv4_d3_kernel"]->data(), weights["conv4_d3_bias"]->data());
	std::cout << "123\n";
	layers["convf5_d3"]->forward(tensors["conv4_d3"]->data(), tensors["convf5_d3"]->data(), weights["convf5_d3_kernel"]->data(), weights["convf5_d3_bias"]->data());
	std::cout << "124\n";
	layers["conv5_d3"]->forward(tensors["convf5_d3"]->data(), tensors["conv5_d3"]->data(), weights["conv5_d3_kernel"]->data(), weights["conv5_d3_bias"]->data());
	std::cout << "128\n";
	layers["convf6_d3"]->forward(tensors["conv5_d3"]->data(), tensors["convf6_d3"]->data(), weights["convf6_d3_kernel"]->data(), weights["convf6_d3_bias"]->data());
	std::cout << "129\n";
	layers["conv6_d3"]->forward(tensors["convf6_d3"]->data(), tensors["conv6_d3"]->data(), weights["conv6_d3_kernel"]->data(), weights["conv6_d3_bias"]->data());
	std::cout << "139\n";
	layers["conv4_d2"]->forward(tensors["Eltwise3"]->data(), tensors["conv4_d2"]->data(), weights["conv4_d2_kernel"]->data(), weights["conv4_d2_bias"]->data());
	std::cout << "143\n";
	layers["convf5_d2"]->forward(tensors["conv4_d2"]->data(), tensors["convf5_d2"]->data(), weights["convf5_d2_kernel"]->data(), weights["convf5_d2_bias"]->data());
	std::cout << "144\n";
	layers["conv5_d2"]->forward(tensors["convf5_d2"]->data(), tensors["conv5_d2"]->data(), weights["conv5_d2_kernel"]->data(), weights["conv5_d2_bias"]->data());
	std::cout << "148\n";
	layers["convf6_d2"]->forward(tensors["conv5_d2"]->data(), tensors["convf6_d2"]->data(), weights["convf6_d2_kernel"]->data(), weights["convf6_d2_bias"]->data());
	std::cout << "149\n";
	layers["conv6_d2"]->forward(tensors["convf6_d2"]->data(), tensors["conv6_d2"]->data(), weights["conv6_d2_kernel"]->data(), weights["conv6_d2_bias"]->data());
	std::cout << "159\n";
	layers["conv4_d1"]->forward(tensors["Eltwise4"]->data(), tensors["conv4_d1"]->data(), weights["conv4_d1_kernel"]->data(), weights["conv4_d1_bias"]->data());
	std::cout << "163\n";
	layers["conv5_d1"]->forward(tensors["conv4_d1"]->data(), tensors["conv5_d1"]->data(), weights["conv5_d1_kernel"]->data(), weights["conv5_d1_bias"]->data());
	std::cout << "167\n";
	layers["conv6_d1"]->forward(tensors["conv5_d1"]->data(), tensors["conv6_d1"]->data(), weights["conv6_d1_kernel"]->data(), weights["conv6_d1_bias"]->data());
	std::cout << "177\n";
	layers["conv4_d0"]->forward(tensors["merge_d3"]->data(), tensors["conv4_d0"]->data(), weights["conv4_d0_kernel"]->data(), weights["conv4_d0_bias"]->data());
	std::cout << "181\n";
	layers["conv5_d0"]->forward(tensors["conv4_d0"]->data(), tensors["conv5_d0"]->data(), weights["conv5_d0_kernel"]->data(), weights["conv5_d0_bias"]->data());
	std::cout << "185\n";
	layers["conv6_d0"]->forward(tensors["conv5_d0"]->data(), tensors["conv6_d0"]->data(), weights["conv6_d0_kernel"]->data(), weights["conv6_d0_bias"]->data());
	std::cout << "190\n";
	layers["conv7_d0"]->forward(tensors["sum4_d0"]->data(), tensors["conv7_d0"]->data(), weights["conv7_d0_kernel"]->data(), weights["conv7_d0_bias"]->data());
	
}

