#include <iostream>
#include <chrono>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	layers["convi"] = znn::phi::jitMakeLayer("conv", "BN=1 IFM=1 OFM=28 ID=18 IHW=196 KD=1 KHW=5 OUT_D_SKIP=0 OUT_PADD=0 OUT_H_SKIP=0 OUT_W_SKIP=0 OUT_PADHW=0 OUT_STRIDE_D=1 OUT_STRIDE_HW=1 ACTIVATION=false ADDOROVERWRITE=false CORES=2 HT=2");
	tensors["convi_kernel"] = new znn::phi::hbw_array<float>(6400);
	tensors["convi_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["convi_kernel"]->data(), weights_path + "convi_kernel.data");
	readArrayFromFile(tensors["convi_bias"]->data(), weights_path + "convi_bias.data");
	layers["crop_layer"] = new znn::phi::CropLayer(1, 28, 18, 192, 28, 14, 180, 2, 6);
	layers["block_input"] = new znn::phi::BlockDataLayer(1, 1, 18, 192);
	layers["convi_padder"] = new znn::phi::PadLayer(1, 1, 18, 192, 0, 2);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(1, 3, 14, 180);
	
	tensors["dummy_data"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 3628800);
	tensors["output"]     = new znn::phi::hbw_array<float>(znn::phi::zero_init, 3628800);

	tensors["user_output"] = new znn::phi::hbw_array<float>(znn::phi::zero_init, 3628800);
	
	input_size = 663552;
	out_dim = 5;
	size_t tmp_shape[] = { 1, 3, 14, 180, 180 };
	out_shape.assign(tmp_shape, tmp_shape + 5);
	size_t tmp_strides[] = { 5443200, 1814400, 129600, 720, 4 };
	out_strides.assign(tmp_strides, tmp_strides + 5);
	
	
}


void znn::phi::Znet::forward(void)
{
	{
	auto begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 1; i++) {
		std::cout << "Starting Forward Pass\n";
		layers["block_input"]->forward(tensors["user_input"]->data(), tensors["input"]->data(), NULL, NULL);
		layers["convi_padder"]->forward(tensors["input"]->data(), tensors["input_padded"]->data(), NULL, NULL);
		layers["convi"]->forward(tensors["input_padded"]->data(), tensors["convi"]->data(), tensors["convi_kernel"]->data(), tensors["convi_bias"]->data(), NULL );
		layers["crop_layer"]->forward(tensors["[u'convi', u'dummy_data']"]->data(), tensors["output"]->data(), NULL, NULL);
		layers["unblock_output"]->forward(tensors["output"]->data(), tensors["user_output"]->data(), NULL, NULL);
		
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	double secs = static_cast<double>(duration) / 1000000;
	std::cout << "average:" << secs/1 << "\n";
	}
	
}

