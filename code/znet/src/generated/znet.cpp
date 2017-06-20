#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["user_output"] = new znn::phi::hbw_array<float>(8);
	tensors["input"] = new znn::phi::hbw_array<float>(32);
	tensors["user_input"] = new znn::phi::hbw_array<float>(32);
	tensors["output"] = new znn::phi::hbw_array<float>(8);
	
	layers["conv_0"] = new znn::phi::ConvWrapper(1, 1, 1, 1, 2, 1, 2, 0, 0, false, 2, 2);
	tensors["conv_0_kernel"] = new znn::phi::hbw_array<float>(256);
	tensors["conv_0_bias"] = new znn::phi::hbw_array<float>(8);
	readArrayFromFile(tensors["conv_0_kernel"]->data(), weights_path + "conv_0_kernel.data");
	readArrayFromFile(tensors["conv_0_bias"]->data(), weights_path + "conv_0_bias.data");
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 1, 1, 1);
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 1, 1, 2);
	
	input_size = 4;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 1, 1, 1, 1 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 4, 4, 4, 4, 4 };
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
			layers["conv_0"]->forward(tensors["input"]->data(), tensors["output"]->data(), tensors["conv_0_kernel"]->data(), tensors["conv_0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv_0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["unblock_output"]->forward(tensors["output"]->data(), tensors["user_output"]->data(), NULL, NULL);
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

