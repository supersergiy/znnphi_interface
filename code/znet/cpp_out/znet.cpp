#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include "znet.hpp"


znn::phi::Znet::Znet(void)
{
	tensors["conv0_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["conv0_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv0_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv0_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv0_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["conv0_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv4_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv4_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv4_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["sum4_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["Deconvolution4"] = new znn::phi::hbw_array<float>(5308416);
	tensors["conv4_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["convf1_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["convf1_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf1_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["convf1_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["Eltwise4"] = new znn::phi::hbw_array<float>(5308416);
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
	tensors["sum0_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["sum0_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["sum0_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["sum0_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf5_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["score"] = new znn::phi::hbw_array<float>(0);
	tensors["Deconvolution3"] = new znn::phi::hbw_array<float>(1990656);
	tensors["convf5_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["convf5_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["input"] = new znn::phi::hbw_array<float>(0);
	tensors["conv2_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["conv2_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv2_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv2_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv2_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["conv2_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv6_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv6_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["sum4_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv6_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["convf6_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["deconv_d3"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv5_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv5_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv5_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv5_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["Deconvolution1"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv5_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["sum4_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["sum4_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["sum4_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv1_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["conv1_d5"] = new znn::phi::hbw_array<float>(62208);
	tensors["conv1_d2"] = new znn::phi::hbw_array<float>(1990656);
	tensors["conv1_d3"] = new znn::phi::hbw_array<float>(663552);
	tensors["conv1_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv1_d1"] = new znn::phi::hbw_array<float>(5308416);
	tensors["convi"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv6_d4"] = new znn::phi::hbw_array<float>(207360);
	tensors["Eltwise3"] = new znn::phi::hbw_array<float>(1990656);
	tensors["Eltwise2"] = new znn::phi::hbw_array<float>(663552);
	tensors["Eltwise1"] = new znn::phi::hbw_array<float>(207360);
	tensors["pool_d2"] = new znn::phi::hbw_array<float>(1327104);
	tensors["pool_d3"] = new znn::phi::hbw_array<float>(497664);
	tensors["pool_d1"] = new znn::phi::hbw_array<float>(3981312);
	tensors["pool_d4"] = new znn::phi::hbw_array<float>(165888);
	tensors["pool_d5"] = new znn::phi::hbw_array<float>(51840);
	tensors["conv7_d0"] = new znn::phi::hbw_array<float>(15925248);
	tensors["output"] = new znn::phi::hbw_array<float>(0);
	tensors["merge_d3"] = new znn::phi::hbw_array<float>(15925248);
	tensors["conv6_d1"] = new znn::phi::hbw_array<float>(5308416);
}


void znn::phi::Znet::forward(void)
{
}

