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
	tensors["user_output"] = new znn::phi::hbw_array<float>(20793600);
	tensors["input"] = new znn::phi::hbw_array<float>(5308416);
	tensors["user_input"] = new znn::phi::hbw_array<float>(5308416);
	tensors["output"] = new znn::phi::hbw_array<float>(20793600);
	
	tensors["conv_kernel"] = new znn::phi::hbw_array<float>(2304);
	tensors["conv_bias"] = new znn::phi::hbw_array<float>(32);
	
	readArrayFromFile(tensors["conv_kernel"]->data(), weights_path + "conv_kernel.data");
	readArrayFromFile(tensors["conv_bias"]->data(), weights_path + "conv_bias.data");
	
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 8, 18, 192);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 28, 18, 190);
	layers["conv"] = new znn::phi::ConvWrapper(1, 8, 28, 18, 192, 1, 3, true, 0, 0);
	
	input_size = 5308416;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 28, 18, 190, 190 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 72777600, 2599200, 144400, 760, 4 };
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
		std::cout << "Running conv!\n";
		layers["conv"]->forward(tensors["input"]->data(), tensors["output"]->data(), tensors["conv_kernel"]->data(), tensors["conv_bias"]->data());
		std::cout << "conv Finished!\n";
		std::cout << "Running unblock_output!\n";
		layers["unblock_output"]->forward(tensors["output"]->data(), tensors["user_output"]->data(), NULL, NULL);
		std::cout << "unblock_output Finished!\n";
		
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	double secs = static_cast<double>(duration) / 1000000;
	std::cout << "average:" << secs/1 << "\n";
}

