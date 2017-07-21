#include <iostream>
#include <chrono>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["conv0_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["user_input"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5308416);
	tensors["conv0_d0_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21678336);
	tensors["conv1_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["convi"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["conv1_d0_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21678336);
	tensors["input_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5531904);
	tensors["output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["user_output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["input"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5308416);
	tensors["convi_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21678336);
	tensors["conv2_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	
	layers["elu0_d0"] = new znn::phi::EluLayer(1, 28, 18, 192);
	layers["conv0_d0"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=28 ID=18 IHW=194 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv0_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv0_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv0_d0_kernel"]->data(), weights_path + "conv0_d0_kernel.data");
	tensors["conv0_d0_bias"]->set_to_const(0);
	layers["elu1_d0"] = new znn::phi::EluLayer(1, 28, 18, 192);
	layers["bn0_d0"] = new znn::phi::ScaleLayer(1, 28, 18, 192);
	tensors["scale_bn0_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn0_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn0_d0"]->data(), weights_path + "scale_bn0_d0.data");
	readArrayFromFile(tensors["bias_bn0_d0"]->data(), weights_path + "bias_bn0_d0.data");
	layers["conv1_d0"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=28 ID=18 IHW=194 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv1_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv1_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv1_d0_kernel"]->data(), weights_path + "conv1_d0_kernel.data");
	tensors["conv1_d0_bias"]->set_to_const(0);
	layers["conv2_d0_padder"] = new znn::phi::PadLayer(1, 28, 18, 192, 0, 1);
	layers["convi"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=1 OFM=28 ID=18 IHW=196 KD=1 KHW=5 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["convi_kernel"] = new znn::phi::hbw_array<float>(6400);
	tensors["convi_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["convi_kernel"]->data(), weights_path + "convi_kernel.data");
	readArrayFromFile(tensors["convi_bias"]->data(), weights_path + "convi_bias.data");
	layers["scale1_d0"] = new znn::phi::ScaleLayer(1, 28, 18, 192);
	tensors["scale_scale1_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale1_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale1_d0"]->data(), weights_path + "scale_scale1_d0.data");
	readArrayFromFile(tensors["bias_scale1_d0"]->data(), weights_path + "bias_scale1_d0.data");
	layers["sum0_d0"] = new znn::phi::EltwiseLayer(1, 28, 18, 192, 1);
	layers["bn1_d0"] = new znn::phi::ScaleLayer(1, 28, 18, 192);
	tensors["scale_bn1_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn1_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn1_d0"]->data(), weights_path + "scale_bn1_d0.data");
	readArrayFromFile(tensors["bias_bn1_d0"]->data(), weights_path + "bias_bn1_d0.data");
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 1, 18, 192);
	layers["conv0_d0_padder"] = new znn::phi::PadLayer(1, 28, 18, 192, 0, 1);
	layers["scale0_d0"] = new znn::phi::ScaleLayer(1, 28, 18, 192);
	tensors["scale_scale0_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale0_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale0_d0"]->data(), weights_path + "scale_scale0_d0.data");
	readArrayFromFile(tensors["bias_scale0_d0"]->data(), weights_path + "bias_scale0_d0.data");
	layers["elui"] = new znn::phi::EluLayer(1, 28, 18, 192);
	layers["conv1_d0_padder"] = new znn::phi::PadLayer(1, 28, 18, 192, 0, 1);
	layers["convi_padder"] = new znn::phi::PadLayer(1, 1, 18, 192, 0, 2);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 28, 18, 192);
	layers["conv2_d0"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=28 ID=18 IHW=194 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv2_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv2_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv2_d0_kernel"]->data(), weights_path + "conv2_d0_kernel.data");
	tensors["conv2_d0_bias"]->set_to_const(0);
	
	input_size = 663552;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 28, 18, 192, 192 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 74317824, 2654208, 147456, 768, 4 };
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
			layers["convi_padder"]->forward(tensors["input"]->data(), tensors["input_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convi_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convi"]->forward(tensors["input_padded"]->data(), tensors["convi"]->data(), tensors["convi_kernel"]->data(), tensors["convi_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convi: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elui"]->forward(tensors["convi"]->data(), tensors["convi"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elui: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d0_padder"]->forward(tensors["convi"]->data(), tensors["convi_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d0_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d0"]->forward(tensors["convi_padded"]->data(), tensors["conv0_d0"]->data(), tensors["conv0_d0_kernel"]->data(), tensors["conv0_d0_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn0_d0"]->forward(tensors["conv0_d0"]->data(), tensors["conv0_d0"]->data(), tensors["scale_bn0_d0"]->data(), tensors["bias_bn0_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn0_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale0_d0"]->forward(tensors["conv0_d0"]->data(), tensors["conv0_d0"]->data(), tensors["scale_scale0_d0"]->data(), tensors["bias_scale0_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale0_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu0_d0"]->forward(tensors["conv0_d0"]->data(), tensors["conv0_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu0_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d0_padder"]->forward(tensors["conv0_d0"]->data(), tensors["conv0_d0_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d0_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d0"]->forward(tensors["conv0_d0_padded"]->data(), tensors["conv1_d0"]->data(), tensors["conv1_d0_kernel"]->data(), tensors["conv1_d0_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn1_d0"]->forward(tensors["conv1_d0"]->data(), tensors["conv1_d0"]->data(), tensors["scale_bn1_d0"]->data(), tensors["bias_bn1_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn1_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale1_d0"]->forward(tensors["conv1_d0"]->data(), tensors["conv1_d0"]->data(), tensors["scale_scale1_d0"]->data(), tensors["bias_scale1_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale1_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu1_d0"]->forward(tensors["conv1_d0"]->data(), tensors["conv1_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu1_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d0_padder"]->forward(tensors["conv1_d0"]->data(), tensors["conv1_d0_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d0_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d0"]->forward(tensors["conv1_d0_padded"]->data(), tensors["conv2_d0"]->data(), tensors["conv2_d0_kernel"]->data(), tensors["conv2_d0_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d0"]->forward(tensors["conv2_d0"]->data(), tensors["output"]->data(), tensors["conv0_d0"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d0: " << secs/1 << "\n";
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

