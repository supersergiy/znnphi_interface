#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include <znn/layer/block_data.hpp>
#include <znn/layer/unblock_data.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["user_output"] = new znn::phi::hbw_array<float>(32);
	tensors["input"] = new znn::phi::hbw_array<float>(32);
	tensors["user_input"] = new znn::phi::hbw_array<float>(32);
	tensors["output"] = new znn::phi::hbw_array<float>(32);
	
	tensors["convi_kernel"] = new znn::phi::hbw_array<float>(64);
	tensors["convi_bias"] = new znn::phi::hbw_array<float>(8);
	
	readArrayFromFile(tensors["convi_kernel"]->data(), weights_path + "convi_kernel.data");
	readArrayFromFile(tensors["convi_bias"]->data(), weights_path + "convi_bias.data");
	
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 1, 1, 2);
	layers["convi"] = new znn::phi::ConvWrapper(1, 1, 8, 1, 2, 1, 1, 0, 0);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 8, 1, 2);
	
	input_size = 4;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 8, 1, 2, 2 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 4, 4, 8, 16, 16 };
	out_strides.assign(tmp_strides, tmp_strides + 5);
	
}


void znn::phi::Znet::forward(void)
{
	auto begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1; i++) {
		std::cout << "Starting Forward Pass\n";
		std::cout << "Running block_input!\n";
		layers["block_input"]->forward(tensors["user_input"]->data(), tensors["input"]->data(), NULL, NULL);
		std::cout << "block_input Finished!\n";
		std::cout << "Running convi!\n";
		layers["convi"]->forward(tensors["input"]->data(), tensors["output"]->data(), tensors["convi_kernel"]->data(), tensors["convi_bias"]->data());
		std::cout << "convi Finished!\n";
		std::cout << "Running unblock_output!\n";
		layers["unblock_output"]->forward(tensors["output"]->data(), tensors["user_output"]->data(), NULL, NULL);
		std::cout << "unblock_output Finished!\n";
		
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	double secs = static_cast<double>(duration) / 1000000;
	std::cout << "average:" << secs/1 << "\n";
}

