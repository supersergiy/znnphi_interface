#include <iostream>
#include <chrono>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["conv0_d1"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 6635520);
	tensors["conv0_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["user_input"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5308416);
	tensors["conv0_d2"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 2160000);
	tensors["conv0_d0_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21678336);
	tensors["convf1_d2"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 2211840);
	tensors["sum0_d1"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 6635520);
	tensors["sum0_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["input_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5531904);
	tensors["pool_d2_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 1800000);
	tensors["user_output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 1990656);
	tensors["input"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5308416);
	tensors["conv2_d1"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 6635520);
	tensors["conv2_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21233664);
	tensors["conv0_d1_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 6914880);
	tensors["pool_d1_padded"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5531904);
	tensors["conv1_d0"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21678336);
	tensors["conv1_d1"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 6914880);
	tensors["convi"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 21678336);
	tensors["pool_d2"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 1658880);
	tensors["pool_d1"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 5308416);
	tensors["output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 1990656);
	
	layers["conv0_d1"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=36 ID=18 IHW=98 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv0_d1_kernel"] = new znn::phi::hbw_array<float>(11520);
	tensors["conv0_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv0_d1_kernel"]->data(), weights_path + "conv0_d1_kernel.data");
	readArrayFromFile(tensors["conv0_d1_bias"]->data(), weights_path + "conv0_d1_bias.data");
	layers["conv0_d0"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=28 ID=18 IHW=194 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv0_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv0_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv0_d0_kernel"]->data(), weights_path + "conv0_d0_kernel.data");
	readArrayFromFile(tensors["conv0_d0_bias"]->data(), weights_path + "conv0_d0_bias.data");
	layers["conv0_d2"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=36 OFM=48 ID=18 IHW=50 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=1 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv0_d2_kernel"] = new znn::phi::hbw_array<float>(17280);
	tensors["conv0_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv0_d2_kernel"]->data(), weights_path + "conv0_d2_kernel.data");
	readArrayFromFile(tensors["conv0_d2_bias"]->data(), weights_path + "conv0_d2_bias.data");
	layers["convf1_d2"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=48 OFM=48 ID=18 IHW=50 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=1 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["convf1_d2_kernel"] = new znn::phi::hbw_array<float>(20736);
	tensors["convf1_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["convf1_d2_kernel"]->data(), weights_path + "convf1_d2_kernel.data");
	tensors["convf1_d2_bias"]->set_to_const(0);
	layers["conv1_d0_padder"] = new znn::phi::PadLayer(1, 28, 18, 192, 0, 1);
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 1, 18, 192);
	layers["conv2_d0"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=28 ID=18 IHW=194 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=true CORES=2 HT=2");
	tensors["conv2_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv2_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv2_d0_kernel"]->data(), weights_path + "conv2_d0_kernel.data");
	readArrayFromFile(tensors["conv2_d0_bias"]->data(), weights_path + "conv2_d0_bias.data");
	tensors["conv2_d0_scale"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv2_d0_scale"]->data(), weights_path + "conv2_d0_scale.data");
	layers["conv0_d2_padder"] = new znn::phi::PadLayer(1, 36, 18, 48, 0, 1);
	layers["conv0_d1_padder"] = new znn::phi::PadLayer(1, 28, 18, 96, 0, 1);
	layers["conv1_d2"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=48 OFM=48 ID=20 IHW=48 KD=3 KHW=1 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv1_d2_kernel"] = new znn::phi::hbw_array<float>(6912);
	tensors["conv1_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv1_d2_kernel"]->data(), weights_path + "conv1_d2_kernel.data");
	tensors["conv1_d2_bias"]->set_to_const(0);
	layers["conv1_d0"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=28 OFM=28 ID=18 IHW=194 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=1 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv1_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv1_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv1_d0_kernel"]->data(), weights_path + "conv1_d0_kernel.data");
	readArrayFromFile(tensors["conv1_d0_bias"]->data(), weights_path + "conv1_d0_bias.data");
	layers["conv1_d1"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=36 OFM=36 ID=18 IHW=98 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=1 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["conv1_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv1_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv1_d1_kernel"]->data(), weights_path + "conv1_d1_kernel.data");
	readArrayFromFile(tensors["conv1_d1_bias"]->data(), weights_path + "conv1_d1_bias.data");
	layers["conv1_d1_padder"] = new znn::phi::PadLayer(1, 36, 18, 96, 0, 1);
	layers["convi"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=1 OFM=28 ID=18 IHW=196 KD=1 KHW=5 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=1 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["convi_kernel"] = new znn::phi::hbw_array<float>(6400);
	tensors["convi_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["convi_kernel"]->data(), weights_path + "convi_kernel.data");
	readArrayFromFile(tensors["convi_bias"]->data(), weights_path + "convi_bias.data");
	layers["pool_d2"] = new znn::phi::MaxPoolingLayer(1, 36, 18, 96, 1, 2, 1, 2);
	layers["pool_d1"] = new znn::phi::MaxPoolingLayer(1, 28, 18, 192, 1, 2, 1, 2);
	layers["conv2_d1"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=36 OFM=36 ID=18 IHW=98 KD=1 KHW=3 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=true ADDOROVERWRITE=true CORES=2 HT=2");
	tensors["conv2_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv2_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv2_d1_kernel"]->data(), weights_path + "conv2_d1_kernel.data");
	readArrayFromFile(tensors["conv2_d1_bias"]->data(), weights_path + "conv2_d1_bias.data");
	tensors["conv2_d1_scale"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv2_d1_scale"]->data(), weights_path + "conv2_d1_scale.data");
	layers["convi_padder"] = new znn::phi::PadLayer(1, 1, 18, 192, 0, 2);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 48, 18, 48);
	
	input_size = 663552;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 48, 18, 48, 48 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 7962624, 165888, 9216, 192, 4 };
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
			layers["conv0_d0"]->forward(tensors["convi"]->data(), tensors["conv0_d0"]->data(), tensors["conv0_d0_kernel"]->data(), tensors["conv0_d0_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d0: " << secs/1 << "\n";
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
			layers["conv2_d0"]->forward(tensors["conv1_d0"]->data(), tensors["conv0_d0"]->data(), tensors["conv2_d0_kernel"]->data(), tensors["conv2_d0_bias"]->data(), tensors["conv2_d0_scale"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d1"]->forward(tensors["conv0_d0"]->data(), tensors["pool_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d1_padder"]->forward(tensors["pool_d1"]->data(), tensors["pool_d1_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d1_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d1"]->forward(tensors["pool_d1_padded"]->data(), tensors["conv0_d1"]->data(), tensors["conv0_d1_kernel"]->data(), tensors["conv0_d1_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d1_padder"]->forward(tensors["conv0_d1"]->data(), tensors["conv0_d1_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d1_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d1"]->forward(tensors["conv0_d1_padded"]->data(), tensors["conv1_d1"]->data(), tensors["conv1_d1_kernel"]->data(), tensors["conv1_d1_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d1"]->forward(tensors["conv1_d1"]->data(), tensors["conv0_d1"]->data(), tensors["conv2_d1_kernel"]->data(), tensors["conv2_d1_bias"]->data(), tensors["conv2_d1_scale"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d2"]->forward(tensors["conv0_d1"]->data(), tensors["pool_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d2_padder"]->forward(tensors["pool_d2"]->data(), tensors["pool_d2_padded"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d2_padder: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d2"]->forward(tensors["pool_d2_padded"]->data(), tensors["conv0_d2"]->data(), tensors["conv0_d2_kernel"]->data(), tensors["conv0_d2_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf1_d2"]->forward(tensors["conv0_d2"]->data(), tensors["convf1_d2"]->data(), tensors["convf1_d2_kernel"]->data(), tensors["convf1_d2_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf1_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d2"]->forward(tensors["convf1_d2"]->data(), tensors["output"]->data(), tensors["conv1_d2_kernel"]->data(), tensors["conv1_d2_bias"]->data(), NULL );
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d2: " << secs/1 << "\n";
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

