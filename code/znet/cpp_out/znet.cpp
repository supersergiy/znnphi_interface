#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include <znn/layer/block_data.hpp>
#include <znn/layer/unblock_data.hpp>
#include <cstring>
#include "znet.hpp"
#include "common.hpp"


znn::phi::Znet::Znet(void)
{
	layers["conv1_d1"] = new znn::phi::ConvWrapper(1, 36, 36, 18, 96, 1, 3, 0, 1);
	
	tensors["conv0_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["conv0_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv0_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv0_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv0_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["conv0_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv4_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv4_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv4_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["user_input"] = new znn::phi::hbw_array<float>(5308416);
	tensors["sum4_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["Deconvolution4"] = new znn::phi::hbw_array<float>(6635520);
	tensors["conv4_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["convf1_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["convf1_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf1_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["convf1_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["Eltwise4"] = new znn::phi::hbw_array<float>(6635520);
	tensors["convf2_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["convf2_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["convf6_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["Deconvolution2"] = new znn::phi::hbw_array<float>(663552);
	tensors["convf6_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["convf2_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf2_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["sum0_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["sum0_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv4_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["sum0_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["sum0_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["sum0_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["sum0_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf5_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["score"] = new znn::phi::hbw_array<float>(5308416);
	tensors["Deconvolution3"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf5_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["convf5_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["user_output"] = new znn::phi::hbw_array<float>(5308416);
	tensors["input"] = new znn::phi::hbw_array<float>(5308416);
	tensors["conv2_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["conv2_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv2_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv2_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv2_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["conv2_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv6_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv6_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["sum4_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv6_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["convf6_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["deconv_d3"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv5_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv5_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv5_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv5_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["Deconvolution1"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv5_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["sum4_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["sum4_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["sum4_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv1_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv1_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["conv1_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv1_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv1_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv1_d1"] = new znn::phi::hbw_array<float>(6635520);
	tensors["convi"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv6_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["Eltwise3"] = new znn::phi::hbw_array<float>(1990656);
	tensors["Eltwise2"] = new znn::phi::hbw_array<float>(663552);
	tensors["Eltwise1"] = new znn::phi::hbw_array<float>(207360);
	tensors["pool_d2"] = new znn::phi::hbw_array<float>(1658880);
	tensors["pool_d3"] = new znn::phi::hbw_array<float>(497664);
	tensors["pool_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["pool_d4"] = new znn::phi::hbw_array<float>(165888);
	tensors["pool_d5"] = new znn::phi::hbw_array<float>(51840);
	tensors["conv7_d0"] = new znn::phi::hbw_array<float>(21233664);
	tensors["output"] = new znn::phi::hbw_array<float>(5308416);
	tensors["merge_d3"] = new znn::phi::hbw_array<float>(21233664);
	tensors["conv6_d1"] = new znn::phi::hbw_array<float>(6635520);
	
	tensors["conv1_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv1_d1_bias"] = new znn::phi::hbw_array<float>(40);
	
	readArrayFromFile(tensors["conv1_d1_kernel"]->data(), "./bin/conv1_d1_kernel.data");
	tensors["conv1_d1_bias"]->set_to_const(0);
	
	
}


void znn::phi::Znet::forward(void)
{
	readArrayFromFile(tensors["user_input"]->data(), "./bin/user_input.data");
	auto begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 20; i++) {
		std::cout << "Starting Forward Pass\n";
		std::cout << "Running conv1_d1!\n";
		layers["conv1_d1"]->forward(tensors["conv0_d1"]->data(), tensors["conv1_d1"]->data(), tensors["conv1_d1_kernel"]->data(), tensors["conv1_d1_bias"]->data());
		std::cout << "conv1_d1 Finished!\n";
		
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	double secs = static_cast<double>(duration) / 1000000;
	std::cout << "average:" << secs/20 << "\n";
}

