#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["user_input"] = new znn::phi::hbw_array<float>(21233664);
	tensors["user_output"] = new znn::phi::hbw_array<float>(20793600);
	tensors["output"] = new znn::phi::hbw_array<float>(20793600);
	tensors["tensor_2"] = new znn::phi::hbw_array<float>(20793600);
	tensors["input"] = new znn::phi::hbw_array<float>(21233664);
	tensors["tensor_0"] = new znn::phi::hbw_array<float>(20793600);
	tensors["tensor_1"] = new znn::phi::hbw_array<float>(20793600);
	
	layers["bnorm_1"] = new znn::phi::ScaleLayer(1, 28, 18, 190);
	tensors["scale_bnorm_1"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bnorm_1"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bnorm_1"]->data(), weights_path + "scale_bnorm_1.data");
	readArrayFromFile(tensors["bias_bnorm_1"]->data(), weights_path + "bias_bnorm_1.data");
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 28, 18, 192);
	layers["scale_2"] = new znn::phi::ScaleLayer(1, 28, 18, 190);
	tensors["scale_scale_2"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale_2"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale_2"]->data(), weights_path + "scale_scale_2.data");
	readArrayFromFile(tensors["bias_scale_2"]->data(), weights_path + "bias_scale_2.data");
	layers["elu_3"] = new znn::phi::EluLayer(1, 28, 18, 190);
	layers["conv_0"] = new znn::phi::ConvWrapper(1, 28, 28, 18, 192, 1, 3, 0, 0, false, 2, 2);
	tensors["conv_0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv_0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv_0_kernel"]->data(), weights_path + "conv_0_kernel.data");
	readArrayFromFile(tensors["conv_0_bias"]->data(), weights_path + "conv_0_bias.data");
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 28, 18, 190);
	
	input_size = 18579456;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 28, 18, 190, 190 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 72777600, 2599200, 144400, 760, 4 };
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
			layers["conv_0"]->forward(tensors["input"]->data(), tensors["tensor_0"]->data(), tensors["conv_0_kernel"]->data(), tensors["conv_0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv_0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bnorm_1"]->forward(tensors["tensor_0"]->data(), tensors["tensor_1"]->data(), tensors["scale_bnorm_1"]->data(), tensors["bias_bnorm_1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bnorm_1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale_2"]->forward(tensors["tensor_1"]->data(), tensors["tensor_2"]->data(), tensors["scale_scale_2"]->data(), tensors["bias_scale_2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale_2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu_3"]->forward(tensors["tensor_2"]->data(), tensors["output"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu_3: " << secs/1 << "\n";
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

