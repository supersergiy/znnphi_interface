#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include <znn/layer/block_data.hpp>
#include <znn/layer/pool/pool.hpp>
#include <znn/layer/unblock_data.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
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
	
	layers["conv0_d1"] = new znn::phi::ConvWrapper(1, 28, 36, 18, 96, 1, 3, 0, 1, true);
	layers["conv0_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["conv0_d3"] = new znn::phi::ConvWrapper(1, 48, 64, 18, 24, 1, 3, 0, 1, true);
	layers["conv0_d2"] = new znn::phi::ConvWrapper(1, 36, 48, 18, 48, 1, 3, 0, 1, true);
	layers["conv0_d5"] = new znn::phi::ConvWrapper(1, 80, 96, 18, 6, 1, 3, 0, 1, true);
	layers["conv0_d4"] = new znn::phi::ConvWrapper(1, 64, 80, 18, 12, 1, 3, 0, 1, true);
	layers["conv4_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0, 1, true);
	layers["conv4_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0, 1, true);
	layers["conv4_d1"] = new znn::phi::ConvWrapper(1, 36, 36, 18, 96, 1, 3, 0, 1, true);
	layers["conv5_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["conv4_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["conv6_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1, 0, true);
	layers["conv6_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1, 0, true);
	layers["convf6_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0, 1, true);
	layers["convf6_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0, 1, true);
	layers["conv6_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1, 0, true);
	layers["convf5_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0, 1, true);
	layers["convf5_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0, 1, true);
	layers["convf5_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0, 1, true);
	layers["conv2_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 3, 1, 1, 0, true);
	layers["conv2_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1, 0, true);
	layers["conv2_d1"] = new znn::phi::ConvWrapper(1, 36, 36, 18, 96, 1, 3, 0, 1, true);
	layers["conv4_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0, 1, true);
	layers["conv6_d1"] = new znn::phi::ConvWrapper(1, 36, 36, 18, 96, 1, 3, 0, 1, true);
	layers["convf6_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0, 1, true);
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 1, 18, 192);
	layers["conv6_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["conv5_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1, 0, true);
	layers["conv5_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1, 0, true);
	layers["conv5_d1"] = new znn::phi::ConvWrapper(1, 36, 36, 18, 96, 1, 3, 0, 1, true);
	layers["conv5_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1, 0, true);
	layers["conv1_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1, 0, true);
	layers["conv1_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 3, 1, 1, 0, true);
	layers["conv1_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1, 0, true);
	layers["conv1_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 3, 1, 1, 0, true);
	layers["conv1_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["conv1_d1"] = new znn::phi::ConvWrapper(1, 36, 36, 18, 96, 1, 3, 0, 1, true);
	layers["convi"] = new znn::phi::ConvWrapper(1, 1, 28, 18, 192, 1, 5, 0, 2, true);
	layers["conv7_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 3, 18, 192);
	layers["convf1_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0, 1, true);
	layers["convf1_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0, 1, true);
	layers["convf1_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 1, 3, 0, 1, true);
	layers["convf1_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0, 1, true);
	layers["convf2_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 1, 3, 0, 1, true);
	layers["convf2_d5"] = new znn::phi::ConvWrapper(1, 96, 96, 18, 6, 1, 3, 0, 1, true);
	layers["convf2_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 1, 3, 0, 1, true);
	layers["convf2_d3"] = new znn::phi::ConvWrapper(1, 64, 64, 18, 24, 1, 3, 0, 1, true);
	layers["conv2_d4"] = new znn::phi::ConvWrapper(1, 80, 80, 18, 12, 3, 1, 1, 0, true);
	layers["conv2_d2"] = new znn::phi::ConvWrapper(1, 48, 48, 18, 48, 3, 1, 1, 0, true);
	layers["conv2_d0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 1, true);
	layers["pool_d2"] = new znn::phi::MaxPoolingLayer(1, 36, 18, 96, 1, 2, 1, 2);
	layers["pool_d3"] = new znn::phi::MaxPoolingLayer(1, 48, 18, 48, 1, 2, 1, 2);
	layers["pool_d1"] = new znn::phi::MaxPoolingLayer(1, 28, 18, 192, 1, 2, 1, 2);
	layers["pool_d4"] = new znn::phi::MaxPoolingLayer(1, 64, 18, 24, 1, 2, 1, 2);
	layers["pool_d5"] = new znn::phi::MaxPoolingLayer(1, 80, 18, 12, 1, 2, 1, 2);
	
	input_size = 663552;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 3, 18, 192, 192 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 7962624, 2654208, 147456, 768, 4 };
	out_strides.assign(tmp_strides, tmp_strides + 5);
	
}


void znn::phi::Znet::forward(void)
{
	{
	auto begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1; i++) {
		std::cout << "Starting Forward Pass\n";
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["block_input"]->forward(tensors["user_input"]->data(), tensors["input"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "block_input" << secs/10 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["pool_d1"]->forward(tensors["sum0_d0"]->data(), tensors["pool_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d1" << secs/10 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["pool_d2"]->forward(tensors["sum0_d1"]->data(), tensors["pool_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d2" << secs/10 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["pool_d3"]->forward(tensors["sum0_d2"]->data(), tensors["pool_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d3" << secs/10 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["pool_d4"]->forward(tensors["sum0_d3"]->data(), tensors["pool_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d4" << secs/10 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["pool_d5"]->forward(tensors["sum0_d4"]->data(), tensors["pool_d5"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d5" << secs/10 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 10; i++) {
			layers["unblock_output"]->forward(tensors["output"]->data(), tensors["user_output"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "unblock_output" << secs/10 << "\n";
		}
		
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	double secs = static_cast<double>(duration) / 1000000;
	std::cout << "average:" << secs/1 << "\n";
	}
}

