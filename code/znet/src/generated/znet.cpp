#include <iostream>
#include <chrono>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["user_input"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 11520);
	tensors["deconv1"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 11520);
	tensors["pooled"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 2880);
	tensors["output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 11520);
	tensors["user_output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 11520);
	tensors["input"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 11520);
	
	layers["pool_1"] = new znn::phi::MaxPoolingLayer(1, 8, 10, 12, 1, 2, 1, 2);
	tensors["deconv_0_kernel"] = new znn::phi::hbw_array<float>(256);
	tensors["deconv_0_bias"] = new znn::phi::hbw_array<float>(8);
	readArrayFromFile(tensors["deconv_0_kernel"]->data(), weights_path + "deconv_0_kernel.data");
	readArrayFromFile(tensors["deconv_0_bias"]->data(), weights_path + "deconv_0_bias.data");
	layers["deconv_0"] = new znn::phi::DeconvAsConvLayer(1, 8, 8, 10, 6, 1, 2, 1, 2, 0, 0, false, true, tensors["deconv_0_kernel"]->data());
	tensors["deconv_0_scale"] = new znn::phi::hbw_array<float>(8);
	readArrayFromFile(tensors["deconv_0_scale"]->data(), weights_path + "deconv_0_scale.data");
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 8, 10, 12);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 8, 10, 12);
	
	input_size = 11520;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 8, 10, 12, 12 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 46080, 5760, 576, 48, 4 };
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
		for (int i = 0; i < 1; i++) {
			layers["block_input"]->forward(tensors["user_input"]->data(), tensors["input"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "block_input: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "input: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_1"]->forward(tensors["input"]->data(), tensors["pooled"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["deconv_0"]->forward(tensors["pooled"]->data(), tensors["input"]->data(), tensors["deconv_0_kernel"]->data(), tensors["deconv_0_bias"]->data(), tensors["deconv_0_scale"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "deconv_0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["unblock_output"]->forward(tensors["input"]->data(), tensors["user_output"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "unblock_output: " << secs/1 << "\n";
		}
		
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	double secs = static_cast<double>(duration) / 1000000;
	std::cout << "average:" << secs/1 << "\n";
	}
	
}

