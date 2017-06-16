#include <iostream>
#include <chrono>
#include <znn/interface/conv_wrapper.hpp>
#include <znn/layer/layers.hpp>
#include <cstring>
#include <znet.hpp>
#include <common.hpp>


znn::phi::Znet::Znet(std::string weights_path)
{
	tensors["conv0_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["conv0_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv0_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv0_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["conv0_d5"] = new znn::phi::hbw_array<float>(124416);
	tensors["conv0_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["conv4_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv4_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["conv4_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["user_input"] = new znn::phi::hbw_array<float>(10616832);
	tensors["sum4_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["Deconvolution4"] = new znn::phi::hbw_array<float>(13271040);
	tensors["conv4_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["convf1_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["convf1_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["convf1_d5"] = new znn::phi::hbw_array<float>(124416);
	tensors["convf1_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["Eltwise4"] = new znn::phi::hbw_array<float>(13271040);
	tensors["convf2_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["convf2_d5"] = new znn::phi::hbw_array<float>(124416);
	tensors["convf6_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["Deconvolution2"] = new znn::phi::hbw_array<float>(1327104);
	tensors["convf6_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["convf2_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["convf2_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["sum0_d5"] = new znn::phi::hbw_array<float>(124416);
	tensors["sum0_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["conv4_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["sum0_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["sum0_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["sum0_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["sum0_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["convf5_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["score"] = new znn::phi::hbw_array<float>(10616832);
	tensors["Deconvolution3"] = new znn::phi::hbw_array<float>(3981312);
	tensors["convf5_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["convf5_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["user_output"] = new znn::phi::hbw_array<float>(10616832);
	tensors["input"] = new znn::phi::hbw_array<float>(10616832);
	tensors["conv2_d5"] = new znn::phi::hbw_array<float>(124416);
	tensors["conv2_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["conv2_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv2_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["conv2_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["conv2_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv6_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv6_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["sum4_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv6_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["convf6_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["deconv_d3"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv5_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["conv5_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv5_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv5_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["Deconvolution1"] = new znn::phi::hbw_array<float>(414720);
	tensors["conv5_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["sum4_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["sum4_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["sum4_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv1_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["conv1_d5"] = new znn::phi::hbw_array<float>(124416);
	tensors["conv1_d2"] = new znn::phi::hbw_array<float>(3981312);
	tensors["conv1_d3"] = new znn::phi::hbw_array<float>(1327104);
	tensors["conv1_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv1_d1"] = new znn::phi::hbw_array<float>(13271040);
	tensors["convi"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv6_d4"] = new znn::phi::hbw_array<float>(414720);
	tensors["Eltwise3"] = new znn::phi::hbw_array<float>(3981312);
	tensors["Eltwise2"] = new znn::phi::hbw_array<float>(1327104);
	tensors["Eltwise1"] = new znn::phi::hbw_array<float>(414720);
	tensors["pool_d2"] = new znn::phi::hbw_array<float>(3317760);
	tensors["pool_d3"] = new znn::phi::hbw_array<float>(995328);
	tensors["pool_d1"] = new znn::phi::hbw_array<float>(10616832);
	tensors["pool_d4"] = new znn::phi::hbw_array<float>(331776);
	tensors["pool_d5"] = new znn::phi::hbw_array<float>(103680);
	tensors["conv7_d0"] = new znn::phi::hbw_array<float>(42467328);
	tensors["output"] = new znn::phi::hbw_array<float>(10616832);
	tensors["merge_d3"] = new znn::phi::hbw_array<float>(42467328);
	tensors["conv6_d1"] = new znn::phi::hbw_array<float>(13271040);
	
	layers["conv0_d1"] = new znn::phi::ConvWrapper(2, 28, 36, 18, 96, 1, 3, 0, 1, false);
	tensors["conv0_d1_kernel"] = new znn::phi::hbw_array<float>(11520);
	tensors["conv0_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv0_d1_kernel"]->data(), weights_path + "conv0_d1_kernel.data");
	tensors["conv0_d1_bias"]->set_to_const(0);
	layers["conv0_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv0_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv0_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv0_d0_kernel"]->data(), weights_path + "conv0_d0_kernel.data");
	tensors["conv0_d0_bias"]->set_to_const(0);
	layers["conv0_d3"] = new znn::phi::ConvWrapper(2, 48, 64, 18, 24, 1, 3, 0, 1, false);
	tensors["conv0_d3_kernel"] = new znn::phi::hbw_array<float>(27648);
	tensors["conv0_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["conv0_d3_kernel"]->data(), weights_path + "conv0_d3_kernel.data");
	tensors["conv0_d3_bias"]->set_to_const(0);
	layers["conv0_d2"] = new znn::phi::ConvWrapper(2, 36, 48, 18, 48, 1, 3, 0, 1, false);
	tensors["conv0_d2_kernel"] = new znn::phi::hbw_array<float>(17280);
	tensors["conv0_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv0_d2_kernel"]->data(), weights_path + "conv0_d2_kernel.data");
	tensors["conv0_d2_bias"]->set_to_const(0);
	layers["conv0_d5"] = new znn::phi::ConvWrapper(2, 80, 96, 18, 6, 1, 3, 0, 1, false);
	tensors["conv0_d5_kernel"] = new znn::phi::hbw_array<float>(69120);
	tensors["conv0_d5_bias"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["conv0_d5_kernel"]->data(), weights_path + "conv0_d5_kernel.data");
	tensors["conv0_d5_bias"]->set_to_const(0);
	layers["conv0_d4"] = new znn::phi::ConvWrapper(2, 64, 80, 18, 12, 1, 3, 0, 1, false);
	tensors["conv0_d4_kernel"] = new znn::phi::hbw_array<float>(46080);
	tensors["conv0_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["conv0_d4_kernel"]->data(), weights_path + "conv0_d4_kernel.data");
	tensors["conv0_d4_bias"]->set_to_const(0);
	layers["conv4_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 1, 3, 0, 1, false);
	tensors["conv4_d3_kernel"] = new znn::phi::hbw_array<float>(36864);
	tensors["conv4_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["conv4_d3_kernel"]->data(), weights_path + "conv4_d3_kernel.data");
	tensors["conv4_d3_bias"]->set_to_const(0);
	layers["conv4_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 1, 3, 0, 1, false);
	tensors["conv4_d4_kernel"] = new znn::phi::hbw_array<float>(57600);
	tensors["conv4_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["conv4_d4_kernel"]->data(), weights_path + "conv4_d4_kernel.data");
	tensors["conv4_d4_bias"]->set_to_const(0);
	layers["scale6_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_scale6_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_scale6_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_scale6_d3"]->data(), weights_path + "scale_scale6_d3.data");
	readArrayFromFile(tensors["bias_scale6_d3"]->data(), weights_path + "bias_scale6_d3.data");
	layers["conv4_d1"] = new znn::phi::ConvWrapper(2, 36, 36, 18, 96, 1, 3, 0, 1, false);
	tensors["conv4_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv4_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv4_d1_kernel"]->data(), weights_path + "conv4_d1_kernel.data");
	tensors["conv4_d1_bias"]->set_to_const(0);
	layers["conv5_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv5_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv5_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv5_d0_kernel"]->data(), weights_path + "conv5_d0_kernel.data");
	tensors["conv5_d0_bias"]->set_to_const(0);
	layers["conv4_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv4_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv4_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv4_d0_kernel"]->data(), weights_path + "conv4_d0_kernel.data");
	tensors["conv4_d0_bias"]->set_to_const(0);
	layers["conv6_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 3, 1, 1, 0, false);
	tensors["conv6_d3_kernel"] = new znn::phi::hbw_array<float>(12288);
	tensors["conv6_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["conv6_d3_kernel"]->data(), weights_path + "conv6_d3_kernel.data");
	tensors["conv6_d3_bias"]->set_to_const(0);
	layers["conv6_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 3, 1, 1, 0, false);
	tensors["conv6_d2_kernel"] = new znn::phi::hbw_array<float>(6912);
	tensors["conv6_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv6_d2_kernel"]->data(), weights_path + "conv6_d2_kernel.data");
	tensors["conv6_d2_bias"]->set_to_const(0);
	layers["convf6_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 1, 3, 0, 1, false);
	tensors["convf6_d2_kernel"] = new znn::phi::hbw_array<float>(20736);
	tensors["convf6_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["convf6_d2_kernel"]->data(), weights_path + "convf6_d2_kernel.data");
	tensors["convf6_d2_bias"]->set_to_const(0);
	layers["elu5_d4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["convf6_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 1, 3, 0, 1, false);
	tensors["convf6_d4_kernel"] = new znn::phi::hbw_array<float>(57600);
	tensors["convf6_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["convf6_d4_kernel"]->data(), weights_path + "convf6_d4_kernel.data");
	tensors["convf6_d4_bias"]->set_to_const(0);
	layers["elu5_d2"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["elu5_d1"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["conv6_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 3, 1, 1, 0, false);
	tensors["conv6_d4_kernel"] = new znn::phi::hbw_array<float>(19200);
	tensors["conv6_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["conv6_d4_kernel"]->data(), weights_path + "conv6_d4_kernel.data");
	tensors["conv6_d4_bias"]->set_to_const(0);
	layers["BatchNorm4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_BatchNorm4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_BatchNorm4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_BatchNorm4"]->data(), weights_path + "scale_BatchNorm4.data");
	readArrayFromFile(tensors["bias_BatchNorm4"]->data(), weights_path + "bias_BatchNorm4.data");
	layers["bn1_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_bn1_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_bn1_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_bn1_d4"]->data(), weights_path + "scale_bn1_d4.data");
	readArrayFromFile(tensors["bias_bn1_d4"]->data(), weights_path + "bias_bn1_d4.data");
	layers["bn1_d5"] = new znn::phi::ScaleLayer(2, 96, 18, 6);
	tensors["scale_bn1_d5"] = new znn::phi::hbw_array<float>(96);
	tensors["bias_bn1_d5"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["scale_bn1_d5"]->data(), weights_path + "scale_bn1_d5.data");
	readArrayFromFile(tensors["bias_bn1_d5"]->data(), weights_path + "bias_bn1_d5.data");
	layers["bn1_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn1_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn1_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn1_d2"]->data(), weights_path + "scale_bn1_d2.data");
	readArrayFromFile(tensors["bias_bn1_d2"]->data(), weights_path + "bias_bn1_d2.data");
	layers["BatchNorm1"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_BatchNorm1"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_BatchNorm1"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_BatchNorm1"]->data(), weights_path + "scale_BatchNorm1.data");
	readArrayFromFile(tensors["bias_BatchNorm1"]->data(), weights_path + "bias_BatchNorm1.data");
	layers["BatchNorm2"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_BatchNorm2"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_BatchNorm2"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_BatchNorm2"]->data(), weights_path + "scale_BatchNorm2.data");
	readArrayFromFile(tensors["bias_BatchNorm2"]->data(), weights_path + "bias_BatchNorm2.data");
	layers["BatchNorm3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_BatchNorm3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_BatchNorm3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_BatchNorm3"]->data(), weights_path + "scale_BatchNorm3.data");
	readArrayFromFile(tensors["bias_BatchNorm3"]->data(), weights_path + "bias_BatchNorm3.data");
	layers["convf5_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 1, 3, 0, 1, false);
	tensors["convf5_d4_kernel"] = new znn::phi::hbw_array<float>(57600);
	tensors["convf5_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["convf5_d4_kernel"]->data(), weights_path + "convf5_d4_kernel.data");
	tensors["convf5_d4_bias"]->set_to_const(0);
	layers["convf5_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 1, 3, 0, 1, false);
	tensors["convf5_d3_kernel"] = new znn::phi::hbw_array<float>(36864);
	tensors["convf5_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["convf5_d3_kernel"]->data(), weights_path + "convf5_d3_kernel.data");
	tensors["convf5_d3_bias"]->set_to_const(0);
	layers["convf5_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 1, 3, 0, 1, false);
	tensors["convf5_d2_kernel"] = new znn::phi::hbw_array<float>(20736);
	tensors["convf5_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["convf5_d2_kernel"]->data(), weights_path + "convf5_d2_kernel.data");
	tensors["convf5_d2_bias"]->set_to_const(0);
	layers["sum4_d1"] = new znn::phi::EltwiseLayer(2, 36, 18, 96, 1);
	layers["bn2_d3"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn2_d3"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn2_d3"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn2_d3"]->data(), weights_path + "scale_bn2_d3.data");
	readArrayFromFile(tensors["bias_bn2_d3"]->data(), weights_path + "bias_bn2_d3.data");
	layers["elu0_d1"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["bn2_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn2_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn2_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn2_d1"]->data(), weights_path + "scale_bn2_d1.data");
	readArrayFromFile(tensors["bias_bn2_d1"]->data(), weights_path + "bias_bn2_d1.data");
	layers["bn3_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_bn3_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_bn3_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_bn3_d3"]->data(), weights_path + "scale_bn3_d3.data");
	readArrayFromFile(tensors["bias_bn3_d3"]->data(), weights_path + "bias_bn3_d3.data");
	layers["sum4_d2"] = new znn::phi::EltwiseLayer(2, 48, 18, 48, 1);
	layers["conv2_d5"] = new znn::phi::ConvWrapper(2, 96, 96, 18, 6, 3, 1, 1, 0, false);
	tensors["conv2_d5_kernel"] = new znn::phi::hbw_array<float>(27648);
	tensors["conv2_d5_bias"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["conv2_d5_kernel"]->data(), weights_path + "conv2_d5_kernel.data");
	tensors["conv2_d5_bias"]->set_to_const(0);
	layers["sum4_d4"] = new znn::phi::EltwiseLayer(2, 80, 18, 12, 1);
	layers["elu0_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["bn2_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn2_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn2_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn2_d2"]->data(), weights_path + "scale_bn2_d2.data");
	readArrayFromFile(tensors["bias_bn2_d2"]->data(), weights_path + "bias_bn2_d2.data");
	layers["elu0_d2"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["bn2_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn2_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn2_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn2_d0"]->data(), weights_path + "scale_bn2_d0.data");
	readArrayFromFile(tensors["bias_bn2_d0"]->data(), weights_path + "bias_bn2_d0.data");
	layers["elu0_d4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["elu0_d5"] = new znn::phi::EluLayer(2, 96, 18, 6);
	layers["bn2_d5"] = new znn::phi::ScaleLayer(2, 96, 18, 6);
	tensors["scale_bn2_d5"] = new znn::phi::hbw_array<float>(96);
	tensors["bias_bn2_d5"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["scale_bn2_d5"]->data(), weights_path + "scale_bn2_d5.data");
	readArrayFromFile(tensors["bias_bn2_d5"]->data(), weights_path + "bias_bn2_d5.data");
	layers["bn2_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_bn2_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_bn2_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_bn2_d4"]->data(), weights_path + "scale_bn2_d4.data");
	readArrayFromFile(tensors["bias_bn2_d4"]->data(), weights_path + "bias_bn2_d4.data");
	layers["conv2_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 3, 1, 1, 0, false);
	tensors["conv2_d3_kernel"] = new znn::phi::hbw_array<float>(12288);
	tensors["conv2_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["conv2_d3_kernel"]->data(), weights_path + "conv2_d3_kernel.data");
	tensors["conv2_d3_bias"]->set_to_const(0);
	layers["elu2_d3"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["conv2_d1"] = new znn::phi::ConvWrapper(2, 36, 36, 18, 96, 1, 3, 0, 1, false);
	tensors["conv2_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv2_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv2_d1_kernel"]->data(), weights_path + "conv2_d1_kernel.data");
	tensors["conv2_d1_bias"]->set_to_const(0);
	layers["elu6_d2"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["elu6_d3"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["elu6_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["elu2_d1"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["scale3_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_scale3_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_scale3_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_scale3_d3"]->data(), weights_path + "scale_scale3_d3.data");
	readArrayFromFile(tensors["bias_scale3_d3"]->data(), weights_path + "bias_scale3_d3.data");
	layers["elu6_d4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["scale1_d3"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale1_d3"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale1_d3"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale1_d3"]->data(), weights_path + "scale_scale1_d3.data");
	readArrayFromFile(tensors["bias_scale1_d3"]->data(), weights_path + "bias_scale1_d3.data");
	layers["scale1_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale1_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale1_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale1_d2"]->data(), weights_path + "scale_scale1_d2.data");
	readArrayFromFile(tensors["bias_scale1_d2"]->data(), weights_path + "bias_scale1_d2.data");
	layers["scale1_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale1_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale1_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale1_d1"]->data(), weights_path + "scale_scale1_d1.data");
	readArrayFromFile(tensors["bias_scale1_d1"]->data(), weights_path + "bias_scale1_d1.data");
	layers["scale1_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale1_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale1_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale1_d0"]->data(), weights_path + "scale_scale1_d0.data");
	readArrayFromFile(tensors["bias_scale1_d0"]->data(), weights_path + "bias_scale1_d0.data");
	layers["scale1_d5"] = new znn::phi::ScaleLayer(2, 96, 18, 6);
	tensors["scale_scale1_d5"] = new znn::phi::hbw_array<float>(96);
	tensors["bias_scale1_d5"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["scale_scale1_d5"]->data(), weights_path + "scale_scale1_d5.data");
	readArrayFromFile(tensors["bias_scale1_d5"]->data(), weights_path + "bias_scale1_d5.data");
	layers["scale1_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_scale1_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_scale1_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_scale1_d4"]->data(), weights_path + "scale_scale1_d4.data");
	readArrayFromFile(tensors["bias_scale1_d4"]->data(), weights_path + "bias_scale1_d4.data");
	layers["bn6_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_bn6_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_bn6_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_bn6_d4"]->data(), weights_path + "scale_bn6_d4.data");
	readArrayFromFile(tensors["bias_bn6_d4"]->data(), weights_path + "bias_bn6_d4.data");
	layers["bn6_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_bn6_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_bn6_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_bn6_d3"]->data(), weights_path + "scale_bn6_d3.data");
	readArrayFromFile(tensors["bias_bn6_d3"]->data(), weights_path + "bias_bn6_d3.data");
	layers["bn6_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn6_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn6_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn6_d2"]->data(), weights_path + "scale_bn6_d2.data");
	readArrayFromFile(tensors["bias_bn6_d2"]->data(), weights_path + "bias_bn6_d2.data");
	layers["bn6_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn6_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn6_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn6_d1"]->data(), weights_path + "scale_bn6_d1.data");
	readArrayFromFile(tensors["bias_bn6_d1"]->data(), weights_path + "bias_bn6_d1.data");
	layers["bn6_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn6_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn6_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn6_d0"]->data(), weights_path + "scale_bn6_d0.data");
	readArrayFromFile(tensors["bias_bn6_d0"]->data(), weights_path + "bias_bn6_d0.data");
	layers["scale0_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_scale0_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_scale0_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_scale0_d4"]->data(), weights_path + "scale_scale0_d4.data");
	readArrayFromFile(tensors["bias_scale0_d4"]->data(), weights_path + "bias_scale0_d4.data");
	layers["scale0_d5"] = new znn::phi::ScaleLayer(2, 96, 18, 6);
	tensors["scale_scale0_d5"] = new znn::phi::hbw_array<float>(96);
	tensors["bias_scale0_d5"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["scale_scale0_d5"]->data(), weights_path + "scale_scale0_d5.data");
	readArrayFromFile(tensors["bias_scale0_d5"]->data(), weights_path + "bias_scale0_d5.data");
	layers["scale0_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale0_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale0_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale0_d0"]->data(), weights_path + "scale_scale0_d0.data");
	readArrayFromFile(tensors["bias_scale0_d0"]->data(), weights_path + "bias_scale0_d0.data");
	layers["scale0_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale0_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale0_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale0_d1"]->data(), weights_path + "scale_scale0_d1.data");
	readArrayFromFile(tensors["bias_scale0_d1"]->data(), weights_path + "bias_scale0_d1.data");
	layers["scale0_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale0_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale0_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale0_d2"]->data(), weights_path + "scale_scale0_d2.data");
	readArrayFromFile(tensors["bias_scale0_d2"]->data(), weights_path + "bias_scale0_d2.data");
	layers["scale0_d3"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale0_d3"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale0_d3"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale0_d3"]->data(), weights_path + "scale_scale0_d3.data");
	readArrayFromFile(tensors["bias_scale0_d3"]->data(), weights_path + "bias_scale0_d3.data");
	layers["bn0_d5"] = new znn::phi::ScaleLayer(2, 96, 18, 6);
	tensors["scale_bn0_d5"] = new znn::phi::hbw_array<float>(96);
	tensors["bias_bn0_d5"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["scale_bn0_d5"]->data(), weights_path + "scale_bn0_d5.data");
	readArrayFromFile(tensors["bias_bn0_d5"]->data(), weights_path + "bias_bn0_d5.data");
	layers["bn0_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_bn0_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_bn0_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_bn0_d4"]->data(), weights_path + "scale_bn0_d4.data");
	readArrayFromFile(tensors["bias_bn0_d4"]->data(), weights_path + "bias_bn0_d4.data");
	layers["bn0_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn0_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn0_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn0_d1"]->data(), weights_path + "scale_bn0_d1.data");
	readArrayFromFile(tensors["bias_bn0_d1"]->data(), weights_path + "bias_bn0_d1.data");
	layers["bn0_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn0_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn0_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn0_d0"]->data(), weights_path + "scale_bn0_d0.data");
	readArrayFromFile(tensors["bias_bn0_d0"]->data(), weights_path + "bias_bn0_d0.data");
	layers["bn0_d3"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn0_d3"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn0_d3"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn0_d3"]->data(), weights_path + "scale_bn0_d3.data");
	readArrayFromFile(tensors["bias_bn0_d3"]->data(), weights_path + "bias_bn0_d3.data");
	layers["bn0_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn0_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn0_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn0_d2"]->data(), weights_path + "scale_bn0_d2.data");
	readArrayFromFile(tensors["bias_bn0_d2"]->data(), weights_path + "bias_bn0_d2.data");
	layers["conv4_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 1, 3, 0, 1, false);
	tensors["conv4_d2_kernel"] = new znn::phi::hbw_array<float>(20736);
	tensors["conv4_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv4_d2_kernel"]->data(), weights_path + "conv4_d2_kernel.data");
	tensors["conv4_d2_bias"]->set_to_const(0);
	layers["conv6_d1"] = new znn::phi::ConvWrapper(2, 36, 36, 18, 96, 1, 3, 0, 1, false);
	tensors["conv6_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv6_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv6_d1_kernel"]->data(), weights_path + "conv6_d1_kernel.data");
	tensors["conv6_d1_bias"]->set_to_const(0);
	layers["convf6_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 1, 3, 0, 1, false);
	tensors["convf6_d3_kernel"] = new znn::phi::hbw_array<float>(36864);
	tensors["convf6_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["convf6_d3_kernel"]->data(), weights_path + "convf6_d3_kernel.data");
	tensors["convf6_d3_bias"]->set_to_const(0);
	layers["bn5_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn5_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn5_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn5_d2"]->data(), weights_path + "scale_bn5_d2.data");
	readArrayFromFile(tensors["bias_bn5_d2"]->data(), weights_path + "bias_bn5_d2.data");
	layers["elu4_d4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["elu5_d3"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["elu4_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["elu4_d1"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["elu4_d2"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["elu4_d3"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["elu6_d1"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["elu5_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["scale4_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale4_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale4_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale4_d0"]->data(), weights_path + "scale_scale4_d0.data");
	readArrayFromFile(tensors["bias_scale4_d0"]->data(), weights_path + "bias_scale4_d0.data");
	layers["scale4_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale4_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale4_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale4_d1"]->data(), weights_path + "scale_scale4_d1.data");
	readArrayFromFile(tensors["bias_scale4_d1"]->data(), weights_path + "bias_scale4_d1.data");
	layers["scale4_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale4_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale4_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale4_d2"]->data(), weights_path + "scale_scale4_d2.data");
	readArrayFromFile(tensors["bias_scale4_d2"]->data(), weights_path + "bias_scale4_d2.data");
	layers["scale4_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_scale4_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_scale4_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_scale4_d3"]->data(), weights_path + "scale_scale4_d3.data");
	readArrayFromFile(tensors["bias_scale4_d3"]->data(), weights_path + "bias_scale4_d3.data");
	layers["scale4_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_scale4_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_scale4_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_scale4_d4"]->data(), weights_path + "scale_scale4_d4.data");
	readArrayFromFile(tensors["bias_scale4_d4"]->data(), weights_path + "bias_scale4_d4.data");
	layers["bn1_d3"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn1_d3"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn1_d3"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn1_d3"]->data(), weights_path + "scale_bn1_d3.data");
	readArrayFromFile(tensors["bias_bn1_d3"]->data(), weights_path + "bias_bn1_d3.data");
	layers["scale2_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale2_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale2_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale2_d2"]->data(), weights_path + "scale_scale2_d2.data");
	readArrayFromFile(tensors["bias_scale2_d2"]->data(), weights_path + "bias_scale2_d2.data");
	layers["scale2_d3"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale2_d3"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale2_d3"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale2_d3"]->data(), weights_path + "scale_scale2_d3.data");
	readArrayFromFile(tensors["bias_scale2_d3"]->data(), weights_path + "bias_scale2_d3.data");
	layers["scale2_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale2_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale2_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale2_d0"]->data(), weights_path + "scale_scale2_d0.data");
	readArrayFromFile(tensors["bias_scale2_d0"]->data(), weights_path + "bias_scale2_d0.data");
	layers["scale2_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale2_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale2_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale2_d1"]->data(), weights_path + "scale_scale2_d1.data");
	readArrayFromFile(tensors["bias_scale2_d1"]->data(), weights_path + "bias_scale2_d1.data");
	layers["bn1_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn1_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn1_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn1_d0"]->data(), weights_path + "scale_bn1_d0.data");
	readArrayFromFile(tensors["bias_bn1_d0"]->data(), weights_path + "bias_bn1_d0.data");
	layers["scale2_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_scale2_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_scale2_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_scale2_d4"]->data(), weights_path + "scale_scale2_d4.data");
	readArrayFromFile(tensors["bias_scale2_d4"]->data(), weights_path + "bias_scale2_d4.data");
	layers["scale2_d5"] = new znn::phi::ScaleLayer(2, 96, 18, 6);
	tensors["scale_scale2_d5"] = new znn::phi::hbw_array<float>(96);
	tensors["bias_scale2_d5"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["scale_scale2_d5"]->data(), weights_path + "scale_scale2_d5.data");
	readArrayFromFile(tensors["bias_scale2_d5"]->data(), weights_path + "bias_scale2_d5.data");
	layers["bn1_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn1_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn1_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn1_d1"]->data(), weights_path + "scale_bn1_d1.data");
	readArrayFromFile(tensors["bias_bn1_d1"]->data(), weights_path + "bias_bn1_d1.data");
	layers["Scale4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_Scale4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_Scale4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_Scale4"]->data(), weights_path + "scale_Scale4.data");
	readArrayFromFile(tensors["bias_Scale4"]->data(), weights_path + "bias_Scale4.data");
	layers["ELU1"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["ELU2"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["ELU3"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["ELU4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["Scale1"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_Scale1"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_Scale1"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_Scale1"]->data(), weights_path + "scale_Scale1.data");
	readArrayFromFile(tensors["bias_Scale1"]->data(), weights_path + "bias_Scale1.data");
	layers["Scale2"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_Scale2"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_Scale2"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_Scale2"]->data(), weights_path + "scale_Scale2.data");
	readArrayFromFile(tensors["bias_Scale2"]->data(), weights_path + "bias_Scale2.data");
	layers["Scale3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_Scale3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_Scale3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_Scale3"]->data(), weights_path + "scale_Scale3.data");
	readArrayFromFile(tensors["bias_Scale3"]->data(), weights_path + "bias_Scale3.data");
	layers["sum0_d5"] = new znn::phi::EltwiseLayer(2, 96, 18, 6, 1);
	layers["sum0_d4"] = new znn::phi::EltwiseLayer(2, 80, 18, 12, 1);
	layers["sum0_d1"] = new znn::phi::EltwiseLayer(2, 36, 18, 96, 1);
	layers["sum0_d0"] = new znn::phi::EltwiseLayer(2, 28, 18, 192, 1);
	layers["sum0_d3"] = new znn::phi::EltwiseLayer(2, 64, 18, 24, 1);
	layers["sum0_d2"] = new znn::phi::EltwiseLayer(2, 48, 18, 48, 1);
	layers["block_input"] = new znn::phi::BlockDataLayer(2, 1, 18, 192);
	layers["conv6_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv6_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv6_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv6_d0_kernel"]->data(), weights_path + "conv6_d0_kernel.data");
	tensors["conv6_d0_bias"]->set_to_const(0);
	layers["elu7_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["sum4_d0"] = new znn::phi::EltwiseLayer(2, 28, 18, 192, 1);
	layers["elu1_d3"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["elu1_d2"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["elu1_d1"] = new znn::phi::EluLayer(2, 36, 18, 96);
	layers["elu1_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["elu1_d5"] = new znn::phi::EluLayer(2, 96, 18, 6);
	layers["elu1_d4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["conv5_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 3, 1, 1, 0, false);
	tensors["conv5_d2_kernel"] = new znn::phi::hbw_array<float>(6912);
	tensors["conv5_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv5_d2_kernel"]->data(), weights_path + "conv5_d2_kernel.data");
	tensors["conv5_d2_bias"]->set_to_const(0);
	layers["conv5_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 3, 1, 1, 0, false);
	tensors["conv5_d3_kernel"] = new znn::phi::hbw_array<float>(12288);
	tensors["conv5_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["conv5_d3_kernel"]->data(), weights_path + "conv5_d3_kernel.data");
	tensors["conv5_d3_bias"]->set_to_const(0);
	layers["elu3_d3"] = new znn::phi::EluLayer(2, 64, 18, 24);
	layers["conv5_d1"] = new znn::phi::ConvWrapper(2, 36, 36, 18, 96, 1, 3, 0, 1, false);
	tensors["conv5_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv5_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv5_d1_kernel"]->data(), weights_path + "conv5_d1_kernel.data");
	tensors["conv5_d1_bias"]->set_to_const(0);
	layers["conv5_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 3, 1, 1, 0, false);
	tensors["conv5_d4_kernel"] = new znn::phi::hbw_array<float>(19200);
	tensors["conv5_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["conv5_d4_kernel"]->data(), weights_path + "conv5_d4_kernel.data");
	tensors["conv5_d4_bias"]->set_to_const(0);
	layers["conv1_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 3, 1, 1, 0, false);
	tensors["conv1_d4_kernel"] = new znn::phi::hbw_array<float>(19200);
	tensors["conv1_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["conv1_d4_kernel"]->data(), weights_path + "conv1_d4_kernel.data");
	tensors["conv1_d4_bias"]->set_to_const(0);
	layers["conv1_d5"] = new znn::phi::ConvWrapper(2, 96, 96, 18, 6, 3, 1, 1, 0, false);
	tensors["conv1_d5_kernel"] = new znn::phi::hbw_array<float>(27648);
	tensors["conv1_d5_bias"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["conv1_d5_kernel"]->data(), weights_path + "conv1_d5_kernel.data");
	tensors["conv1_d5_bias"]->set_to_const(0);
	layers["conv1_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 3, 1, 1, 0, false);
	tensors["conv1_d2_kernel"] = new znn::phi::hbw_array<float>(6912);
	tensors["conv1_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv1_d2_kernel"]->data(), weights_path + "conv1_d2_kernel.data");
	tensors["conv1_d2_bias"]->set_to_const(0);
	layers["conv1_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 3, 1, 1, 0, false);
	tensors["conv1_d3_kernel"] = new znn::phi::hbw_array<float>(12288);
	tensors["conv1_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["conv1_d3_kernel"]->data(), weights_path + "conv1_d3_kernel.data");
	tensors["conv1_d3_bias"]->set_to_const(0);
	layers["conv1_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv1_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv1_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv1_d0_kernel"]->data(), weights_path + "conv1_d0_kernel.data");
	tensors["conv1_d0_bias"]->set_to_const(0);
	layers["conv1_d1"] = new znn::phi::ConvWrapper(2, 36, 36, 18, 96, 1, 3, 0, 1, false);
	tensors["conv1_d1_kernel"] = new znn::phi::hbw_array<float>(14400);
	tensors["conv1_d1_bias"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["conv1_d1_kernel"]->data(), weights_path + "conv1_d1_kernel.data");
	tensors["conv1_d1_bias"]->set_to_const(0);
	layers["convi"] = new znn::phi::ConvWrapper(2, 1, 28, 18, 192, 1, 5, 0, 2, false);
	tensors["convi_kernel"] = new znn::phi::hbw_array<float>(6400);
	tensors["convi_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["convi_kernel"]->data(), weights_path + "convi_kernel.data");
	readArrayFromFile(tensors["convi_bias"]->data(), weights_path + "convi_bias.data");
	layers["conv7_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv7_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv7_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv7_d0_kernel"]->data(), weights_path + "conv7_d0_kernel.data");
	readArrayFromFile(tensors["conv7_d0_bias"]->data(), weights_path + "conv7_d0_bias.data");
	layers["merge_d3"] = new znn::phi::EltwiseLayer(2, 28, 18, 192, 1);
	layers["unblock_output"] = new znn::phi::UnblockDataLayer(2, 3, 18, 192);
	layers["bn4_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn4_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn4_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn4_d0"]->data(), weights_path + "scale_bn4_d0.data");
	readArrayFromFile(tensors["bias_bn4_d0"]->data(), weights_path + "bias_bn4_d0.data");
	layers["scale5_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_scale5_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_scale5_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_scale5_d4"]->data(), weights_path + "scale_scale5_d4.data");
	readArrayFromFile(tensors["bias_scale5_d4"]->data(), weights_path + "bias_scale5_d4.data");
	layers["scale5_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_scale5_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_scale5_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_scale5_d3"]->data(), weights_path + "scale_scale5_d3.data");
	readArrayFromFile(tensors["bias_scale5_d3"]->data(), weights_path + "bias_scale5_d3.data");
	layers["scale5_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale5_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale5_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale5_d2"]->data(), weights_path + "scale_scale5_d2.data");
	readArrayFromFile(tensors["bias_scale5_d2"]->data(), weights_path + "bias_scale5_d2.data");
	layers["scale5_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale5_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale5_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale5_d1"]->data(), weights_path + "scale_scale5_d1.data");
	readArrayFromFile(tensors["bias_scale5_d1"]->data(), weights_path + "bias_scale5_d1.data");
	layers["scale5_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale5_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale5_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale5_d0"]->data(), weights_path + "scale_scale5_d0.data");
	readArrayFromFile(tensors["bias_scale5_d0"]->data(), weights_path + "bias_scale5_d0.data");
	layers["convf1_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 1, 3, 0, 1, false);
	tensors["convf1_d3_kernel"] = new znn::phi::hbw_array<float>(36864);
	tensors["convf1_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["convf1_d3_kernel"]->data(), weights_path + "convf1_d3_kernel.data");
	tensors["convf1_d3_bias"]->set_to_const(0);
	layers["convf1_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 1, 3, 0, 1, false);
	tensors["convf1_d2_kernel"] = new znn::phi::hbw_array<float>(20736);
	tensors["convf1_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["convf1_d2_kernel"]->data(), weights_path + "convf1_d2_kernel.data");
	tensors["convf1_d2_bias"]->set_to_const(0);
	layers["convf1_d5"] = new znn::phi::ConvWrapper(2, 96, 96, 18, 6, 1, 3, 0, 1, false);
	tensors["convf1_d5_kernel"] = new znn::phi::hbw_array<float>(82944);
	tensors["convf1_d5_bias"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["convf1_d5_kernel"]->data(), weights_path + "convf1_d5_kernel.data");
	tensors["convf1_d5_bias"]->set_to_const(0);
	layers["convf1_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 1, 3, 0, 1, false);
	tensors["convf1_d4_kernel"] = new znn::phi::hbw_array<float>(57600);
	tensors["convf1_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["convf1_d4_kernel"]->data(), weights_path + "convf1_d4_kernel.data");
	tensors["convf1_d4_bias"]->set_to_const(0);
	layers["convf2_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 1, 3, 0, 1, false);
	tensors["convf2_d4_kernel"] = new znn::phi::hbw_array<float>(57600);
	tensors["convf2_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["convf2_d4_kernel"]->data(), weights_path + "convf2_d4_kernel.data");
	tensors["convf2_d4_bias"]->set_to_const(0);
	layers["convf2_d5"] = new znn::phi::ConvWrapper(2, 96, 96, 18, 6, 1, 3, 0, 1, false);
	tensors["convf2_d5_kernel"] = new znn::phi::hbw_array<float>(82944);
	tensors["convf2_d5_bias"] = new znn::phi::hbw_array<float>(96);
	readArrayFromFile(tensors["convf2_d5_kernel"]->data(), weights_path + "convf2_d5_kernel.data");
	tensors["convf2_d5_bias"]->set_to_const(0);
	layers["convf2_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 1, 3, 0, 1, false);
	tensors["convf2_d2_kernel"] = new znn::phi::hbw_array<float>(20736);
	tensors["convf2_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["convf2_d2_kernel"]->data(), weights_path + "convf2_d2_kernel.data");
	tensors["convf2_d2_bias"]->set_to_const(0);
	layers["convf2_d3"] = new znn::phi::ConvWrapper(2, 64, 64, 18, 24, 1, 3, 0, 1, false);
	tensors["convf2_d3_kernel"] = new znn::phi::hbw_array<float>(36864);
	tensors["convf2_d3_bias"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["convf2_d3_kernel"]->data(), weights_path + "convf2_d3_kernel.data");
	tensors["convf2_d3_bias"]->set_to_const(0);
	layers["bn4_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn4_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn4_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn4_d1"]->data(), weights_path + "scale_bn4_d1.data");
	readArrayFromFile(tensors["bias_bn4_d1"]->data(), weights_path + "bias_bn4_d1.data");
	layers["sum4_d3"] = new znn::phi::EltwiseLayer(2, 64, 18, 24, 1);
	layers["scale6_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_scale6_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_scale6_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_scale6_d4"]->data(), weights_path + "scale_scale6_d4.data");
	readArrayFromFile(tensors["bias_scale6_d4"]->data(), weights_path + "bias_scale6_d4.data");
	layers["bn4_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_bn4_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_bn4_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_bn4_d2"]->data(), weights_path + "scale_bn4_d2.data");
	readArrayFromFile(tensors["bias_bn4_d2"]->data(), weights_path + "bias_bn4_d2.data");
	layers["scale6_d2"] = new znn::phi::ScaleLayer(2, 48, 18, 48);
	tensors["scale_scale6_d2"] = new znn::phi::hbw_array<float>(48);
	tensors["bias_scale6_d2"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["scale_scale6_d2"]->data(), weights_path + "scale_scale6_d2.data");
	readArrayFromFile(tensors["bias_scale6_d2"]->data(), weights_path + "bias_scale6_d2.data");
	layers["bn4_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_bn4_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_bn4_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_bn4_d4"]->data(), weights_path + "scale_bn4_d4.data");
	readArrayFromFile(tensors["bias_bn4_d4"]->data(), weights_path + "bias_bn4_d4.data");
	layers["scale6_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_scale6_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_scale6_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_scale6_d0"]->data(), weights_path + "scale_scale6_d0.data");
	readArrayFromFile(tensors["bias_scale6_d0"]->data(), weights_path + "bias_scale6_d0.data");
	layers["scale6_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_scale6_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_scale6_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_scale6_d1"]->data(), weights_path + "scale_scale6_d1.data");
	readArrayFromFile(tensors["bias_scale6_d1"]->data(), weights_path + "bias_scale6_d1.data");
	layers["bn5_d1"] = new znn::phi::ScaleLayer(2, 36, 18, 96);
	tensors["scale_bn5_d1"] = new znn::phi::hbw_array<float>(40);
	tensors["bias_bn5_d1"] = new znn::phi::hbw_array<float>(40);
	readArrayFromFile(tensors["scale_bn5_d1"]->data(), weights_path + "scale_bn5_d1.data");
	readArrayFromFile(tensors["bias_bn5_d1"]->data(), weights_path + "bias_bn5_d1.data");
	layers["elu2_d4"] = new znn::phi::EluLayer(2, 80, 18, 12);
	layers["conv2_d4"] = new znn::phi::ConvWrapper(2, 80, 80, 18, 12, 3, 1, 1, 0, false);
	tensors["conv2_d4_kernel"] = new znn::phi::hbw_array<float>(19200);
	tensors["conv2_d4_bias"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["conv2_d4_kernel"]->data(), weights_path + "conv2_d4_kernel.data");
	tensors["conv2_d4_bias"]->set_to_const(0);
	layers["elu2_d2"] = new znn::phi::EluLayer(2, 48, 18, 48);
	layers["conv2_d2"] = new znn::phi::ConvWrapper(2, 48, 48, 18, 48, 3, 1, 1, 0, false);
	tensors["conv2_d2_kernel"] = new znn::phi::hbw_array<float>(6912);
	tensors["conv2_d2_bias"] = new znn::phi::hbw_array<float>(48);
	readArrayFromFile(tensors["conv2_d2_kernel"]->data(), weights_path + "conv2_d2_kernel.data");
	tensors["conv2_d2_bias"]->set_to_const(0);
	layers["elu2_d0"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["conv2_d0"] = new znn::phi::ConvWrapper(2, 28, 28, 18, 192, 1, 3, 0, 1, false);
	tensors["conv2_d0_kernel"] = new znn::phi::hbw_array<float>(9216);
	tensors["conv2_d0_bias"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["conv2_d0_kernel"]->data(), weights_path + "conv2_d0_kernel.data");
	tensors["conv2_d0_bias"]->set_to_const(0);
	layers["elu2_d5"] = new znn::phi::EluLayer(2, 96, 18, 6);
	layers["bn5_d4"] = new znn::phi::ScaleLayer(2, 80, 18, 12);
	tensors["scale_bn5_d4"] = new znn::phi::hbw_array<float>(80);
	tensors["bias_bn5_d4"] = new znn::phi::hbw_array<float>(80);
	readArrayFromFile(tensors["scale_bn5_d4"]->data(), weights_path + "scale_bn5_d4.data");
	readArrayFromFile(tensors["bias_bn5_d4"]->data(), weights_path + "bias_bn5_d4.data");
	layers["elui"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["Eltwise4"] = new znn::phi::EltwiseLayer(2, 36, 18, 96, 1);
	layers["Eltwise3"] = new znn::phi::EltwiseLayer(2, 48, 18, 48, 1);
	layers["Eltwise2"] = new znn::phi::EltwiseLayer(2, 64, 18, 24, 1);
	layers["Eltwise1"] = new znn::phi::EltwiseLayer(2, 80, 18, 12, 1);
	layers["pool_d2"] = new znn::phi::MaxPoolingLayer(2, 36, 18, 96, 1, 2, 1, 2);
	layers["pool_d3"] = new znn::phi::MaxPoolingLayer(2, 48, 18, 48, 1, 2, 1, 2);
	layers["bn5_d0"] = new znn::phi::ScaleLayer(2, 28, 18, 192);
	tensors["scale_bn5_d0"] = new znn::phi::hbw_array<float>(32);
	tensors["bias_bn5_d0"] = new znn::phi::hbw_array<float>(32);
	readArrayFromFile(tensors["scale_bn5_d0"]->data(), weights_path + "scale_bn5_d0.data");
	readArrayFromFile(tensors["bias_bn5_d0"]->data(), weights_path + "bias_bn5_d0.data");
	layers["pool_d1"] = new znn::phi::MaxPoolingLayer(2, 28, 18, 192, 1, 2, 1, 2);
	layers["bn5_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_bn5_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_bn5_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_bn5_d3"]->data(), weights_path + "scale_bn5_d3.data");
	readArrayFromFile(tensors["bias_bn5_d3"]->data(), weights_path + "bias_bn5_d3.data");
	layers["pool_d4"] = new znn::phi::MaxPoolingLayer(2, 64, 18, 24, 1, 2, 1, 2);
	layers["pool_d5"] = new znn::phi::MaxPoolingLayer(2, 80, 18, 12, 1, 2, 1, 2);
	layers["elu0_d3"] = new znn::phi::EluLayer(2, 28, 18, 192);
	layers["bn4_d3"] = new znn::phi::ScaleLayer(2, 64, 18, 24);
	tensors["scale_bn4_d3"] = new znn::phi::hbw_array<float>(64);
	tensors["bias_bn4_d3"] = new znn::phi::hbw_array<float>(64);
	readArrayFromFile(tensors["scale_bn4_d3"]->data(), weights_path + "scale_bn4_d3.data");
	readArrayFromFile(tensors["bias_bn4_d3"]->data(), weights_path + "bias_bn4_d3.data");
	
	input_size = 1327104;
	out_dim = 5;
	size_t tmp_shape[] = { 2, 3, 18, 192, 192 };
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
			layers["convi"]->forward(tensors["input"]->data(), tensors["convi"]->data(), tensors["convi_kernel"]->data(), tensors["convi_bias"]->data());
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
			layers["conv0_d0"]->forward(tensors["convi"]->data(), tensors["conv0_d0"]->data(), tensors["conv0_d0_kernel"]->data(), tensors["conv0_d0_bias"]->data());
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
			layers["conv1_d0"]->forward(tensors["conv0_d0"]->data(), tensors["conv1_d0"]->data(), tensors["conv1_d0_kernel"]->data(), tensors["conv1_d0_bias"]->data());
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
			layers["conv2_d0"]->forward(tensors["conv1_d0"]->data(), tensors["conv2_d0"]->data(), tensors["conv2_d0_kernel"]->data(), tensors["conv2_d0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d0"]->forward(tensors["conv2_d0"]->data(), tensors["sum0_d0"]->data(), tensors["conv0_d0"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn2_d0"]->forward(tensors["sum0_d0"]->data(), tensors["sum0_d0"]->data(), tensors["scale_bn2_d0"]->data(), tensors["bias_bn2_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn2_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale2_d0"]->forward(tensors["sum0_d0"]->data(), tensors["sum0_d0"]->data(), tensors["scale_scale2_d0"]->data(), tensors["bias_scale2_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale2_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu2_d0"]->forward(tensors["sum0_d0"]->data(), tensors["sum0_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu2_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d1"]->forward(tensors["sum0_d0"]->data(), tensors["pool_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d1"]->forward(tensors["pool_d1"]->data(), tensors["conv0_d1"]->data(), tensors["conv0_d1_kernel"]->data(), tensors["conv0_d1_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn0_d1"]->forward(tensors["conv0_d1"]->data(), tensors["conv0_d1"]->data(), tensors["scale_bn0_d1"]->data(), tensors["bias_bn0_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn0_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale0_d1"]->forward(tensors["conv0_d1"]->data(), tensors["conv0_d1"]->data(), tensors["scale_scale0_d1"]->data(), tensors["bias_scale0_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale0_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu0_d1"]->forward(tensors["conv0_d1"]->data(), tensors["conv0_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu0_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d1"]->forward(tensors["conv0_d1"]->data(), tensors["conv1_d1"]->data(), tensors["conv1_d1_kernel"]->data(), tensors["conv1_d1_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn1_d1"]->forward(tensors["conv1_d1"]->data(), tensors["conv1_d1"]->data(), tensors["scale_bn1_d1"]->data(), tensors["bias_bn1_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn1_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale1_d1"]->forward(tensors["conv1_d1"]->data(), tensors["conv1_d1"]->data(), tensors["scale_scale1_d1"]->data(), tensors["bias_scale1_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale1_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu1_d1"]->forward(tensors["conv1_d1"]->data(), tensors["conv1_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu1_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d1"]->forward(tensors["conv1_d1"]->data(), tensors["conv2_d1"]->data(), tensors["conv2_d1_kernel"]->data(), tensors["conv2_d1_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d1"]->forward(tensors["conv2_d1"]->data(), tensors["sum0_d1"]->data(), tensors["conv0_d1"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn2_d1"]->forward(tensors["sum0_d1"]->data(), tensors["sum0_d1"]->data(), tensors["scale_bn2_d1"]->data(), tensors["bias_bn2_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn2_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale2_d1"]->forward(tensors["sum0_d1"]->data(), tensors["sum0_d1"]->data(), tensors["scale_scale2_d1"]->data(), tensors["bias_scale2_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale2_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu2_d1"]->forward(tensors["sum0_d1"]->data(), tensors["sum0_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu2_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d2"]->forward(tensors["sum0_d1"]->data(), tensors["pool_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d2"]->forward(tensors["pool_d2"]->data(), tensors["conv0_d2"]->data(), tensors["conv0_d2_kernel"]->data(), tensors["conv0_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn0_d2"]->forward(tensors["conv0_d2"]->data(), tensors["conv0_d2"]->data(), tensors["scale_bn0_d2"]->data(), tensors["bias_bn0_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn0_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale0_d2"]->forward(tensors["conv0_d2"]->data(), tensors["conv0_d2"]->data(), tensors["scale_scale0_d2"]->data(), tensors["bias_scale0_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale0_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu0_d2"]->forward(tensors["conv0_d2"]->data(), tensors["conv0_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu0_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf1_d2"]->forward(tensors["conv0_d2"]->data(), tensors["convf1_d2"]->data(), tensors["convf1_d2_kernel"]->data(), tensors["convf1_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf1_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d2"]->forward(tensors["convf1_d2"]->data(), tensors["conv1_d2"]->data(), tensors["conv1_d2_kernel"]->data(), tensors["conv1_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn1_d2"]->forward(tensors["conv1_d2"]->data(), tensors["conv1_d2"]->data(), tensors["scale_bn1_d2"]->data(), tensors["bias_bn1_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn1_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale1_d2"]->forward(tensors["conv1_d2"]->data(), tensors["conv1_d2"]->data(), tensors["scale_scale1_d2"]->data(), tensors["bias_scale1_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale1_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu1_d2"]->forward(tensors["conv1_d2"]->data(), tensors["conv1_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu1_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf2_d2"]->forward(tensors["conv1_d2"]->data(), tensors["convf2_d2"]->data(), tensors["convf2_d2_kernel"]->data(), tensors["convf2_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf2_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d2"]->forward(tensors["convf2_d2"]->data(), tensors["conv2_d2"]->data(), tensors["conv2_d2_kernel"]->data(), tensors["conv2_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d2"]->forward(tensors["conv2_d2"]->data(), tensors["sum0_d2"]->data(), tensors["conv0_d2"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn2_d2"]->forward(tensors["sum0_d2"]->data(), tensors["sum0_d2"]->data(), tensors["scale_bn2_d2"]->data(), tensors["bias_bn2_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn2_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale2_d2"]->forward(tensors["sum0_d2"]->data(), tensors["sum0_d2"]->data(), tensors["scale_scale2_d2"]->data(), tensors["bias_scale2_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale2_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu2_d2"]->forward(tensors["sum0_d2"]->data(), tensors["sum0_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu2_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d3"]->forward(tensors["sum0_d2"]->data(), tensors["pool_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d3"]->forward(tensors["pool_d3"]->data(), tensors["conv0_d3"]->data(), tensors["conv0_d3_kernel"]->data(), tensors["conv0_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["BatchNorm1"]->forward(tensors["conv0_d3"]->data(), tensors["conv0_d3"]->data(), tensors["scale_BatchNorm1"]->data(), tensors["bias_BatchNorm1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "BatchNorm1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Scale1"]->forward(tensors["conv0_d3"]->data(), tensors["conv0_d3"]->data(), tensors["scale_Scale1"]->data(), tensors["bias_Scale1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Scale1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["ELU1"]->forward(tensors["conv0_d3"]->data(), tensors["conv0_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "ELU1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf1_d3"]->forward(tensors["conv0_d3"]->data(), tensors["convf1_d3"]->data(), tensors["convf1_d3_kernel"]->data(), tensors["convf1_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf1_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d3"]->forward(tensors["convf1_d3"]->data(), tensors["conv1_d3"]->data(), tensors["conv1_d3_kernel"]->data(), tensors["conv1_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["BatchNorm2"]->forward(tensors["conv1_d3"]->data(), tensors["conv1_d3"]->data(), tensors["scale_BatchNorm2"]->data(), tensors["bias_BatchNorm2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "BatchNorm2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Scale2"]->forward(tensors["conv1_d3"]->data(), tensors["conv1_d3"]->data(), tensors["scale_Scale2"]->data(), tensors["bias_Scale2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Scale2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["ELU2"]->forward(tensors["conv1_d3"]->data(), tensors["conv1_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "ELU2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf2_d3"]->forward(tensors["conv1_d3"]->data(), tensors["convf2_d3"]->data(), tensors["convf2_d3_kernel"]->data(), tensors["convf2_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf2_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d3"]->forward(tensors["convf2_d3"]->data(), tensors["conv2_d3"]->data(), tensors["conv2_d3_kernel"]->data(), tensors["conv2_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d3"]->forward(tensors["conv2_d3"]->data(), tensors["sum0_d3"]->data(), tensors["conv0_d3"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["BatchNorm3"]->forward(tensors["sum0_d3"]->data(), tensors["sum0_d3"]->data(), tensors["scale_BatchNorm3"]->data(), tensors["bias_BatchNorm3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "BatchNorm3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Scale3"]->forward(tensors["sum0_d3"]->data(), tensors["sum0_d3"]->data(), tensors["scale_Scale3"]->data(), tensors["bias_Scale3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Scale3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["ELU3"]->forward(tensors["sum0_d3"]->data(), tensors["sum0_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "ELU3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d4"]->forward(tensors["sum0_d3"]->data(), tensors["pool_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d4"]->forward(tensors["pool_d4"]->data(), tensors["conv0_d4"]->data(), tensors["conv0_d4_kernel"]->data(), tensors["conv0_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn0_d4"]->forward(tensors["conv0_d4"]->data(), tensors["conv0_d4"]->data(), tensors["scale_bn0_d4"]->data(), tensors["bias_bn0_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn0_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale0_d4"]->forward(tensors["conv0_d4"]->data(), tensors["conv0_d4"]->data(), tensors["scale_scale0_d4"]->data(), tensors["bias_scale0_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale0_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu0_d4"]->forward(tensors["conv0_d4"]->data(), tensors["conv0_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu0_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf1_d4"]->forward(tensors["conv0_d4"]->data(), tensors["convf1_d4"]->data(), tensors["convf1_d4_kernel"]->data(), tensors["convf1_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf1_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d4"]->forward(tensors["convf1_d4"]->data(), tensors["conv1_d4"]->data(), tensors["conv1_d4_kernel"]->data(), tensors["conv1_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn1_d4"]->forward(tensors["conv1_d4"]->data(), tensors["conv1_d4"]->data(), tensors["scale_bn1_d4"]->data(), tensors["bias_bn1_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn1_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale1_d4"]->forward(tensors["conv1_d4"]->data(), tensors["conv1_d4"]->data(), tensors["scale_scale1_d4"]->data(), tensors["bias_scale1_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale1_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu1_d4"]->forward(tensors["conv1_d4"]->data(), tensors["conv1_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu1_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf2_d4"]->forward(tensors["conv1_d4"]->data(), tensors["convf2_d4"]->data(), tensors["convf2_d4_kernel"]->data(), tensors["convf2_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf2_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d4"]->forward(tensors["convf2_d4"]->data(), tensors["conv2_d4"]->data(), tensors["conv2_d4_kernel"]->data(), tensors["conv2_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d4"]->forward(tensors["conv2_d4"]->data(), tensors["sum0_d4"]->data(), tensors["conv0_d4"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn2_d4"]->forward(tensors["sum0_d4"]->data(), tensors["sum0_d4"]->data(), tensors["scale_bn2_d4"]->data(), tensors["bias_bn2_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn2_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale2_d4"]->forward(tensors["sum0_d4"]->data(), tensors["sum0_d4"]->data(), tensors["scale_scale2_d4"]->data(), tensors["bias_scale2_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale2_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu2_d4"]->forward(tensors["sum0_d4"]->data(), tensors["sum0_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu2_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["pool_d5"]->forward(tensors["sum0_d4"]->data(), tensors["pool_d5"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "pool_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv0_d5"]->forward(tensors["pool_d5"]->data(), tensors["conv0_d5"]->data(), tensors["conv0_d5_kernel"]->data(), tensors["conv0_d5_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv0_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn0_d5"]->forward(tensors["conv0_d5"]->data(), tensors["conv0_d5"]->data(), tensors["scale_bn0_d5"]->data(), tensors["bias_bn0_d5"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn0_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale0_d5"]->forward(tensors["conv0_d5"]->data(), tensors["conv0_d5"]->data(), tensors["scale_scale0_d5"]->data(), tensors["bias_scale0_d5"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale0_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu0_d5"]->forward(tensors["conv0_d5"]->data(), tensors["conv0_d5"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu0_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf1_d5"]->forward(tensors["conv0_d5"]->data(), tensors["convf1_d5"]->data(), tensors["convf1_d5_kernel"]->data(), tensors["convf1_d5_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf1_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv1_d5"]->forward(tensors["convf1_d5"]->data(), tensors["conv1_d5"]->data(), tensors["conv1_d5_kernel"]->data(), tensors["conv1_d5_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv1_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn1_d5"]->forward(tensors["conv1_d5"]->data(), tensors["conv1_d5"]->data(), tensors["scale_bn1_d5"]->data(), tensors["bias_bn1_d5"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn1_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale1_d5"]->forward(tensors["conv1_d5"]->data(), tensors["conv1_d5"]->data(), tensors["scale_scale1_d5"]->data(), tensors["bias_scale1_d5"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale1_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu1_d5"]->forward(tensors["conv1_d5"]->data(), tensors["conv1_d5"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu1_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf2_d5"]->forward(tensors["conv1_d5"]->data(), tensors["convf2_d5"]->data(), tensors["convf2_d5_kernel"]->data(), tensors["convf2_d5_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf2_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv2_d5"]->forward(tensors["convf2_d5"]->data(), tensors["conv2_d5"]->data(), tensors["conv2_d5_kernel"]->data(), tensors["conv2_d5_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv2_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum0_d5"]->forward(tensors["conv2_d5"]->data(), tensors["sum0_d5"]->data(), tensors["conv0_d5"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum0_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn2_d5"]->forward(tensors["sum0_d5"]->data(), tensors["sum0_d5"]->data(), tensors["scale_bn2_d5"]->data(), tensors["bias_bn2_d5"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn2_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale2_d5"]->forward(tensors["sum0_d5"]->data(), tensors["sum0_d5"]->data(), tensors["scale_scale2_d5"]->data(), tensors["bias_scale2_d5"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale2_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu2_d5"]->forward(tensors["sum0_d5"]->data(), tensors["sum0_d5"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu2_d5: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Deconvolution1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Eltwise1"]->forward(tensors["Deconvolution1"]->data(), tensors["Eltwise1"]->data(), tensors["sum0_d4"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Eltwise1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["BatchNorm4"]->forward(tensors["Eltwise1"]->data(), tensors["Eltwise1"]->data(), tensors["scale_BatchNorm4"]->data(), tensors["bias_BatchNorm4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "BatchNorm4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Scale4"]->forward(tensors["Eltwise1"]->data(), tensors["Eltwise1"]->data(), tensors["scale_Scale4"]->data(), tensors["bias_Scale4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Scale4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["ELU4"]->forward(tensors["Eltwise1"]->data(), tensors["Eltwise1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "ELU4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv4_d4"]->forward(tensors["Eltwise1"]->data(), tensors["conv4_d4"]->data(), tensors["conv4_d4_kernel"]->data(), tensors["conv4_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv4_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn4_d4"]->forward(tensors["conv4_d4"]->data(), tensors["conv4_d4"]->data(), tensors["scale_bn4_d4"]->data(), tensors["bias_bn4_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn4_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale4_d4"]->forward(tensors["conv4_d4"]->data(), tensors["conv4_d4"]->data(), tensors["scale_scale4_d4"]->data(), tensors["bias_scale4_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale4_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu4_d4"]->forward(tensors["conv4_d4"]->data(), tensors["conv4_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu4_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf5_d4"]->forward(tensors["conv4_d4"]->data(), tensors["convf5_d4"]->data(), tensors["convf5_d4_kernel"]->data(), tensors["convf5_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf5_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv5_d4"]->forward(tensors["convf5_d4"]->data(), tensors["conv5_d4"]->data(), tensors["conv5_d4_kernel"]->data(), tensors["conv5_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv5_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn5_d4"]->forward(tensors["conv5_d4"]->data(), tensors["conv5_d4"]->data(), tensors["scale_bn5_d4"]->data(), tensors["bias_bn5_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn5_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale5_d4"]->forward(tensors["conv5_d4"]->data(), tensors["conv5_d4"]->data(), tensors["scale_scale5_d4"]->data(), tensors["bias_scale5_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale5_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu5_d4"]->forward(tensors["conv5_d4"]->data(), tensors["conv5_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu5_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf6_d4"]->forward(tensors["conv5_d4"]->data(), tensors["convf6_d4"]->data(), tensors["convf6_d4_kernel"]->data(), tensors["convf6_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf6_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv6_d4"]->forward(tensors["convf6_d4"]->data(), tensors["conv6_d4"]->data(), tensors["conv6_d4_kernel"]->data(), tensors["conv6_d4_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv6_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum4_d4"]->forward(tensors["conv6_d4"]->data(), tensors["sum4_d4"]->data(), tensors["conv4_d4"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum4_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn6_d4"]->forward(tensors["sum4_d4"]->data(), tensors["sum4_d4"]->data(), tensors["scale_bn6_d4"]->data(), tensors["bias_bn6_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn6_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale6_d4"]->forward(tensors["sum4_d4"]->data(), tensors["sum4_d4"]->data(), tensors["scale_scale6_d4"]->data(), tensors["bias_scale6_d4"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale6_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu6_d4"]->forward(tensors["sum4_d4"]->data(), tensors["sum4_d4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu6_d4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Deconvolution2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Eltwise2"]->forward(tensors["Deconvolution2"]->data(), tensors["Eltwise2"]->data(), tensors["sum0_d3"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Eltwise2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn3_d3"]->forward(tensors["Eltwise2"]->data(), tensors["Eltwise2"]->data(), tensors["scale_bn3_d3"]->data(), tensors["bias_bn3_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn3_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale3_d3"]->forward(tensors["Eltwise2"]->data(), tensors["Eltwise2"]->data(), tensors["scale_scale3_d3"]->data(), tensors["bias_scale3_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale3_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu3_d3"]->forward(tensors["Eltwise2"]->data(), tensors["Eltwise2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu3_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv4_d3"]->forward(tensors["Eltwise2"]->data(), tensors["conv4_d3"]->data(), tensors["conv4_d3_kernel"]->data(), tensors["conv4_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv4_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn4_d3"]->forward(tensors["conv4_d3"]->data(), tensors["conv4_d3"]->data(), tensors["scale_bn4_d3"]->data(), tensors["bias_bn4_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn4_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale4_d3"]->forward(tensors["conv4_d3"]->data(), tensors["conv4_d3"]->data(), tensors["scale_scale4_d3"]->data(), tensors["bias_scale4_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale4_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu4_d3"]->forward(tensors["conv4_d3"]->data(), tensors["conv4_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu4_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf5_d3"]->forward(tensors["conv4_d3"]->data(), tensors["convf5_d3"]->data(), tensors["convf5_d3_kernel"]->data(), tensors["convf5_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf5_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv5_d3"]->forward(tensors["convf5_d3"]->data(), tensors["conv5_d3"]->data(), tensors["conv5_d3_kernel"]->data(), tensors["conv5_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv5_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn5_d3"]->forward(tensors["conv5_d3"]->data(), tensors["conv5_d3"]->data(), tensors["scale_bn5_d3"]->data(), tensors["bias_bn5_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn5_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale5_d3"]->forward(tensors["conv5_d3"]->data(), tensors["conv5_d3"]->data(), tensors["scale_scale5_d3"]->data(), tensors["bias_scale5_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale5_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu5_d3"]->forward(tensors["conv5_d3"]->data(), tensors["conv5_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu5_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf6_d3"]->forward(tensors["conv5_d3"]->data(), tensors["convf6_d3"]->data(), tensors["convf6_d3_kernel"]->data(), tensors["convf6_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf6_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv6_d3"]->forward(tensors["convf6_d3"]->data(), tensors["conv6_d3"]->data(), tensors["conv6_d3_kernel"]->data(), tensors["conv6_d3_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv6_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum4_d3"]->forward(tensors["conv6_d3"]->data(), tensors["sum4_d3"]->data(), tensors["conv4_d3"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum4_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn6_d3"]->forward(tensors["sum4_d3"]->data(), tensors["sum4_d3"]->data(), tensors["scale_bn6_d3"]->data(), tensors["bias_bn6_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn6_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale6_d3"]->forward(tensors["sum4_d3"]->data(), tensors["sum4_d3"]->data(), tensors["scale_scale6_d3"]->data(), tensors["bias_scale6_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale6_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu6_d3"]->forward(tensors["sum4_d3"]->data(), tensors["sum4_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu6_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Deconvolution3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Eltwise3"]->forward(tensors["Deconvolution3"]->data(), tensors["Eltwise3"]->data(), tensors["sum0_d2"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Eltwise3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn2_d3"]->forward(tensors["Eltwise3"]->data(), tensors["Eltwise3"]->data(), tensors["scale_bn2_d3"]->data(), tensors["bias_bn2_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn2_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale2_d3"]->forward(tensors["Eltwise3"]->data(), tensors["Eltwise3"]->data(), tensors["scale_scale2_d3"]->data(), tensors["bias_scale2_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale2_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu2_d3"]->forward(tensors["Eltwise3"]->data(), tensors["Eltwise3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu2_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv4_d2"]->forward(tensors["Eltwise3"]->data(), tensors["conv4_d2"]->data(), tensors["conv4_d2_kernel"]->data(), tensors["conv4_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv4_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn4_d2"]->forward(tensors["conv4_d2"]->data(), tensors["conv4_d2"]->data(), tensors["scale_bn4_d2"]->data(), tensors["bias_bn4_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn4_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale4_d2"]->forward(tensors["conv4_d2"]->data(), tensors["conv4_d2"]->data(), tensors["scale_scale4_d2"]->data(), tensors["bias_scale4_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale4_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu4_d2"]->forward(tensors["conv4_d2"]->data(), tensors["conv4_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu4_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf5_d2"]->forward(tensors["conv4_d2"]->data(), tensors["convf5_d2"]->data(), tensors["convf5_d2_kernel"]->data(), tensors["convf5_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf5_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv5_d2"]->forward(tensors["convf5_d2"]->data(), tensors["conv5_d2"]->data(), tensors["conv5_d2_kernel"]->data(), tensors["conv5_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv5_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn5_d2"]->forward(tensors["conv5_d2"]->data(), tensors["conv5_d2"]->data(), tensors["scale_bn5_d2"]->data(), tensors["bias_bn5_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn5_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale5_d2"]->forward(tensors["conv5_d2"]->data(), tensors["conv5_d2"]->data(), tensors["scale_scale5_d2"]->data(), tensors["bias_scale5_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale5_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu5_d2"]->forward(tensors["conv5_d2"]->data(), tensors["conv5_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu5_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["convf6_d2"]->forward(tensors["conv5_d2"]->data(), tensors["convf6_d2"]->data(), tensors["convf6_d2_kernel"]->data(), tensors["convf6_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "convf6_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv6_d2"]->forward(tensors["convf6_d2"]->data(), tensors["conv6_d2"]->data(), tensors["conv6_d2_kernel"]->data(), tensors["conv6_d2_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv6_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum4_d2"]->forward(tensors["conv6_d2"]->data(), tensors["sum4_d2"]->data(), tensors["conv4_d2"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum4_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn6_d2"]->forward(tensors["sum4_d2"]->data(), tensors["sum4_d2"]->data(), tensors["scale_bn6_d2"]->data(), tensors["bias_bn6_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn6_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale6_d2"]->forward(tensors["sum4_d2"]->data(), tensors["sum4_d2"]->data(), tensors["scale_scale6_d2"]->data(), tensors["bias_scale6_d2"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale6_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu6_d2"]->forward(tensors["sum4_d2"]->data(), tensors["sum4_d2"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu6_d2: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Deconvolution4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["Eltwise4"]->forward(tensors["Deconvolution4"]->data(), tensors["Eltwise4"]->data(), tensors["sum0_d1"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "Eltwise4: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn1_d3"]->forward(tensors["Eltwise4"]->data(), tensors["Eltwise4"]->data(), tensors["scale_bn1_d3"]->data(), tensors["bias_bn1_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn1_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale1_d3"]->forward(tensors["Eltwise4"]->data(), tensors["Eltwise4"]->data(), tensors["scale_scale1_d3"]->data(), tensors["bias_scale1_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale1_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu1_d3"]->forward(tensors["Eltwise4"]->data(), tensors["Eltwise4"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu1_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv4_d1"]->forward(tensors["Eltwise4"]->data(), tensors["conv4_d1"]->data(), tensors["conv4_d1_kernel"]->data(), tensors["conv4_d1_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv4_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn4_d1"]->forward(tensors["conv4_d1"]->data(), tensors["conv4_d1"]->data(), tensors["scale_bn4_d1"]->data(), tensors["bias_bn4_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn4_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale4_d1"]->forward(tensors["conv4_d1"]->data(), tensors["conv4_d1"]->data(), tensors["scale_scale4_d1"]->data(), tensors["bias_scale4_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale4_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu4_d1"]->forward(tensors["conv4_d1"]->data(), tensors["conv4_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu4_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv5_d1"]->forward(tensors["conv4_d1"]->data(), tensors["conv5_d1"]->data(), tensors["conv5_d1_kernel"]->data(), tensors["conv5_d1_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv5_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn5_d1"]->forward(tensors["conv5_d1"]->data(), tensors["conv5_d1"]->data(), tensors["scale_bn5_d1"]->data(), tensors["bias_bn5_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn5_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale5_d1"]->forward(tensors["conv5_d1"]->data(), tensors["conv5_d1"]->data(), tensors["scale_scale5_d1"]->data(), tensors["bias_scale5_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale5_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu5_d1"]->forward(tensors["conv5_d1"]->data(), tensors["conv5_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu5_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv6_d1"]->forward(tensors["conv5_d1"]->data(), tensors["conv6_d1"]->data(), tensors["conv6_d1_kernel"]->data(), tensors["conv6_d1_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv6_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum4_d1"]->forward(tensors["conv6_d1"]->data(), tensors["sum4_d1"]->data(), tensors["conv4_d1"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum4_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn6_d1"]->forward(tensors["sum4_d1"]->data(), tensors["sum4_d1"]->data(), tensors["scale_bn6_d1"]->data(), tensors["bias_bn6_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn6_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale6_d1"]->forward(tensors["sum4_d1"]->data(), tensors["sum4_d1"]->data(), tensors["scale_scale6_d1"]->data(), tensors["bias_scale6_d1"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale6_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu6_d1"]->forward(tensors["sum4_d1"]->data(), tensors["sum4_d1"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu6_d1: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "deconv_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["merge_d3"]->forward(tensors["deconv_d3"]->data(), tensors["merge_d3"]->data(), tensors["sum0_d0"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "merge_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn0_d3"]->forward(tensors["merge_d3"]->data(), tensors["merge_d3"]->data(), tensors["scale_bn0_d3"]->data(), tensors["bias_bn0_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn0_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale0_d3"]->forward(tensors["merge_d3"]->data(), tensors["merge_d3"]->data(), tensors["scale_scale0_d3"]->data(), tensors["bias_scale0_d3"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale0_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu0_d3"]->forward(tensors["merge_d3"]->data(), tensors["merge_d3"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu0_d3: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv4_d0"]->forward(tensors["merge_d3"]->data(), tensors["conv4_d0"]->data(), tensors["conv4_d0_kernel"]->data(), tensors["conv4_d0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv4_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn4_d0"]->forward(tensors["conv4_d0"]->data(), tensors["conv4_d0"]->data(), tensors["scale_bn4_d0"]->data(), tensors["bias_bn4_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn4_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale4_d0"]->forward(tensors["conv4_d0"]->data(), tensors["conv4_d0"]->data(), tensors["scale_scale4_d0"]->data(), tensors["bias_scale4_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale4_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu4_d0"]->forward(tensors["conv4_d0"]->data(), tensors["conv4_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu4_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv5_d0"]->forward(tensors["conv4_d0"]->data(), tensors["conv5_d0"]->data(), tensors["conv5_d0_kernel"]->data(), tensors["conv5_d0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv5_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn5_d0"]->forward(tensors["conv5_d0"]->data(), tensors["conv5_d0"]->data(), tensors["scale_bn5_d0"]->data(), tensors["bias_bn5_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn5_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale5_d0"]->forward(tensors["conv5_d0"]->data(), tensors["conv5_d0"]->data(), tensors["scale_scale5_d0"]->data(), tensors["bias_scale5_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale5_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu5_d0"]->forward(tensors["conv5_d0"]->data(), tensors["conv5_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu5_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv6_d0"]->forward(tensors["conv5_d0"]->data(), tensors["conv6_d0"]->data(), tensors["conv6_d0_kernel"]->data(), tensors["conv6_d0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv6_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["sum4_d0"]->forward(tensors["conv6_d0"]->data(), tensors["sum4_d0"]->data(), tensors["conv4_d0"]->data(), NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "sum4_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["bn6_d0"]->forward(tensors["sum4_d0"]->data(), tensors["sum4_d0"]->data(), tensors["scale_bn6_d0"]->data(), tensors["bias_bn6_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "bn6_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["scale6_d0"]->forward(tensors["sum4_d0"]->data(), tensors["sum4_d0"]->data(), tensors["scale_scale6_d0"]->data(), tensors["bias_scale6_d0"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "scale6_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu6_d0"]->forward(tensors["sum4_d0"]->data(), tensors["sum4_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu6_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["conv7_d0"]->forward(tensors["sum4_d0"]->data(), tensors["conv7_d0"]->data(), tensors["conv7_d0_kernel"]->data(), tensors["conv7_d0_bias"]->data());
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "conv7_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
			layers["elu7_d0"]->forward(tensors["conv7_d0"]->data(), tensors["conv7_d0"]->data(), NULL, NULL);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "elu7_d0: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "score: " << secs/1 << "\n";
		}
		{
		auto begin = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1; i++) {
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
		double secs = static_cast<double>(duration) / 1000000;
		std::cout << "output: " << secs/1 << "\n";
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

