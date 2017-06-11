#include <znn/interface/conv_wrapper.hpp>
#include <znn/tensor/tensor.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>


int main(void) {
    znn::phi::hbw_array<float> tensor_sum4_d1(442368);
    znn::phi::hbw_array<float> tensor_sum4_d4(11520);
    znn::phi::hbw_array<float> tensor_sum4_d3(36864);
    znn::phi::hbw_array<float> tensor_Deconvolution3(147456);
    znn::phi::hbw_array<float> tensor_conv4_d1(442368);
    znn::phi::hbw_array<float> tensor_convf2_d2(552960);
    znn::phi::hbw_array<float> tensor_conv2_d3(110592);
    znn::phi::hbw_array<float> tensor_conv4_d2(110592);
    znn::phi::hbw_array<float> tensor_sum0_d5(3456);
    znn::phi::hbw_array<float> tensor_conv5_d0(1179648);
    znn::phi::hbw_array<float> tensor_convf6_d3(36864);
    znn::phi::hbw_array<float> tensor_input(10616832);
    znn::phi::hbw_array<float> tensor_sum0_d2(552960);
    znn::phi::hbw_array<float> tensor_conv2_d4(23040);
    znn::phi::hbw_array<float> tensor_Deconvolution4(442368);
    znn::phi::hbw_array<float> tensor_Deconvolution1(13824);
    znn::phi::hbw_array<float> tensor_conv6_d0(1179648);
    znn::phi::hbw_array<float> tensor_conv1_d1(3981312);
    znn::phi::hbw_array<float> tensor_conv2_d0(21233664);
    znn::phi::hbw_array<float> tensor_Eltwise2(46080);
    znn::phi::hbw_array<float> tensor_conv0_d2(552960);
    znn::phi::hbw_array<float> tensor_conv5_d4(11520);
    znn::phi::hbw_array<float> tensor_merge_d3(1769472);
    znn::phi::hbw_array<float> tensor_conv1_d3(110592);
    znn::phi::hbw_array<float> tensor_conv2_d2(552960);
    znn::phi::hbw_array<float> tensor_conv1_d0(21233664);
    znn::phi::hbw_array<float> tensor_sum0_d0(21233664);
    znn::phi::hbw_array<float> tensor_convf1_d5(3456);
    znn::phi::hbw_array<float> tensor_score(4718592);
    znn::phi::hbw_array<float> tensor_convf5_d3(36864);
    znn::phi::hbw_array<float> tensor_conv5_d2(110592);
    znn::phi::hbw_array<float> tensor_conv7_d0(1179648);
    znn::phi::hbw_array<float> tensor_Eltwise4(442368);
    znn::phi::hbw_array<float> tensor_convi(21233664);
    znn::phi::hbw_array<float> tensor_conv1_d4(23040);
    znn::phi::hbw_array<float> tensor_deconv_d3(1769472);
    znn::phi::hbw_array<float> tensor_convf2_d3(110592);
    znn::phi::hbw_array<float> tensor_conv5_d1(442368);
    znn::phi::hbw_array<float> tensor_conv1_d2(552960);
    znn::phi::hbw_array<float> tensor_pool_d5(2880);
    znn::phi::hbw_array<float> tensor_conv6_d4(11520);
    znn::phi::hbw_array<float> tensor_convf6_d4(11520);
    znn::phi::hbw_array<float> tensor_convf2_d5(3456);
    znn::phi::hbw_array<float> tensor_sum0_d3(110592);
    znn::phi::hbw_array<float> tensor_pool_d4(18432);
    znn::phi::hbw_array<float> tensor_conv4_d3(36864);
    znn::phi::hbw_array<float> tensor_convf1_d3(110592);
    znn::phi::hbw_array<float> tensor_convf5_d4(11520);
    znn::phi::hbw_array<float> tensor_pool_d3(82944);
    znn::phi::hbw_array<float> tensor_Eltwise1(13824);
    znn::phi::hbw_array<float> tensor_conv0_d4(23040);
    znn::phi::hbw_array<float> tensor_Deconvolution2(46080);
    znn::phi::hbw_array<float> tensor_conv4_d4(11520);
    znn::phi::hbw_array<float> tensor_conv2_d1(3981312);
    znn::phi::hbw_array<float> tensor_conv4_d0(1179648);
    znn::phi::hbw_array<float> tensor_conv0_d1(3981312);
    znn::phi::hbw_array<float> tensor_convf5_d2(110592);
    znn::phi::hbw_array<float> tensor_conv0_d0(21233664);
    znn::phi::hbw_array<float> tensor_pool_d1(2654208);
    znn::phi::hbw_array<float> tensor_convf1_d2(552960);
    znn::phi::hbw_array<float> tensor_conv6_d2(110592);
    znn::phi::hbw_array<float> tensor_sum4_d2(110592);
    znn::phi::hbw_array<float> tensor_conv0_d3(110592);
    znn::phi::hbw_array<float> tensor_convf6_d2(110592);
    znn::phi::hbw_array<float> tensor_sum0_d4(23040);
    znn::phi::hbw_array<float> tensor_pool_d2(552960);
    znn::phi::hbw_array<float> tensor_conv1_d5(3456);
    znn::phi::hbw_array<float> tensor_conv5_d3(36864);
    znn::phi::hbw_array<float> tensor_conv2_d5(3456);
    znn::phi::hbw_array<float> tensor_sum0_d1(3981312);
    znn::phi::hbw_array<float> tensor_convf1_d4(23040);
    znn::phi::hbw_array<float> tensor_convf2_d4(23040);
    znn::phi::hbw_array<float> tensor_Eltwise3(147456);
    znn::phi::hbw_array<float> tensor_conv6_d1(442368);
    znn::phi::hbw_array<float> tensor_sum4_d0(1179648);
    znn::phi::hbw_array<float> tensor_conv6_d3(36864);
    znn::phi::hbw_array<float> tensor_conv0_d5(3456);
    znn::phi::hbw_array<float> tensor_convi_kernel(12800);
    znn::phi::hbw_array<float> tensor_convi_bias(32);
    znn::phi::hbw_array<float> tensor_conv0_d0_kernel(9216);
    znn::phi::hbw_array<float> tensor_conv0_d0_bias(32);
    znn::phi::hbw_array<float> tensor_conv1_d0_kernel(9216);
    znn::phi::hbw_array<float> tensor_conv1_d0_bias(32);
    znn::phi::hbw_array<float> tensor_conv2_d0_kernel(9216);
    znn::phi::hbw_array<float> tensor_conv2_d0_bias(32);
    znn::phi::hbw_array<float> tensor_conv0_d1_kernel(13824);
    znn::phi::hbw_array<float> tensor_conv0_d1_bias(48);
    znn::phi::hbw_array<float> tensor_conv1_d1_kernel(20736);
    znn::phi::hbw_array<float> tensor_conv1_d1_bias(48);
    znn::phi::hbw_array<float> tensor_conv2_d1_kernel(20736);
    znn::phi::hbw_array<float> tensor_conv2_d1_bias(48);
    znn::phi::hbw_array<float> tensor_conv0_d2_kernel(20736);
    znn::phi::hbw_array<float> tensor_conv0_d2_bias(48);
    znn::phi::hbw_array<float> tensor_convf1_d2_kernel(20736);
    znn::phi::hbw_array<float> tensor_convf1_d2_bias(48);
    znn::phi::hbw_array<float> tensor_conv1_d2_kernel(6912);
    znn::phi::hbw_array<float> tensor_conv1_d2_bias(48);
    znn::phi::hbw_array<float> tensor_convf2_d2_kernel(20736);
    znn::phi::hbw_array<float> tensor_convf2_d2_bias(48);
    znn::phi::hbw_array<float> tensor_conv2_d2_kernel(6912);
    znn::phi::hbw_array<float> tensor_conv2_d2_bias(48);
    znn::phi::hbw_array<float> tensor_conv0_d3_kernel(27648);
    znn::phi::hbw_array<float> tensor_conv0_d3_bias(64);
    znn::phi::hbw_array<float> tensor_convf1_d3_kernel(36864);
    znn::phi::hbw_array<float> tensor_convf1_d3_bias(64);
    znn::phi::hbw_array<float> tensor_conv1_d3_kernel(12288);
    znn::phi::hbw_array<float> tensor_conv1_d3_bias(64);
    znn::phi::hbw_array<float> tensor_convf2_d3_kernel(36864);
    znn::phi::hbw_array<float> tensor_convf2_d3_bias(64);
    znn::phi::hbw_array<float> tensor_conv2_d3_kernel(12288);
    znn::phi::hbw_array<float> tensor_conv2_d3_bias(64);
    znn::phi::hbw_array<float> tensor_conv0_d4_kernel(46080);
    znn::phi::hbw_array<float> tensor_conv0_d4_bias(80);
    znn::phi::hbw_array<float> tensor_convf1_d4_kernel(57600);
    znn::phi::hbw_array<float> tensor_convf1_d4_bias(80);
    znn::phi::hbw_array<float> tensor_conv1_d4_kernel(19200);
    znn::phi::hbw_array<float> tensor_conv1_d4_bias(80);
    znn::phi::hbw_array<float> tensor_convf2_d4_kernel(57600);
    znn::phi::hbw_array<float> tensor_convf2_d4_bias(80);
    znn::phi::hbw_array<float> tensor_conv2_d4_kernel(19200);
    znn::phi::hbw_array<float> tensor_conv2_d4_bias(80);
    znn::phi::hbw_array<float> tensor_conv0_d5_kernel(69120);
    znn::phi::hbw_array<float> tensor_conv0_d5_bias(96);
    znn::phi::hbw_array<float> tensor_convf1_d5_kernel(82944);
    znn::phi::hbw_array<float> tensor_convf1_d5_bias(96);
    znn::phi::hbw_array<float> tensor_conv1_d5_kernel(27648);
    znn::phi::hbw_array<float> tensor_conv1_d5_bias(96);
    znn::phi::hbw_array<float> tensor_convf2_d5_kernel(82944);
    znn::phi::hbw_array<float> tensor_convf2_d5_bias(96);
    znn::phi::hbw_array<float> tensor_conv2_d5_kernel(27648);
    znn::phi::hbw_array<float> tensor_conv2_d5_bias(96);
    znn::phi::hbw_array<float> tensor_conv4_d4_kernel(69120);
    znn::phi::hbw_array<float> tensor_conv4_d4_bias(80);
    znn::phi::hbw_array<float> tensor_convf5_d4_kernel(57600);
    znn::phi::hbw_array<float> tensor_convf5_d4_bias(80);
    znn::phi::hbw_array<float> tensor_conv5_d4_kernel(19200);
    znn::phi::hbw_array<float> tensor_conv5_d4_bias(80);
    znn::phi::hbw_array<float> tensor_convf6_d4_kernel(57600);
    znn::phi::hbw_array<float> tensor_convf6_d4_bias(80);
    znn::phi::hbw_array<float> tensor_conv6_d4_kernel(19200);
    znn::phi::hbw_array<float> tensor_conv6_d4_bias(80);
    znn::phi::hbw_array<float> tensor_conv4_d3_kernel(46080);
    znn::phi::hbw_array<float> tensor_conv4_d3_bias(64);
    znn::phi::hbw_array<float> tensor_convf5_d3_kernel(36864);
    znn::phi::hbw_array<float> tensor_convf5_d3_bias(64);
    znn::phi::hbw_array<float> tensor_conv5_d3_kernel(12288);
    znn::phi::hbw_array<float> tensor_conv5_d3_bias(64);
    znn::phi::hbw_array<float> tensor_convf6_d3_kernel(36864);
    znn::phi::hbw_array<float> tensor_convf6_d3_bias(64);
    znn::phi::hbw_array<float> tensor_conv6_d3_kernel(12288);
    znn::phi::hbw_array<float> tensor_conv6_d3_bias(64);
    znn::phi::hbw_array<float> tensor_conv4_d2_kernel(27648);
    znn::phi::hbw_array<float> tensor_conv4_d2_bias(48);
    znn::phi::hbw_array<float> tensor_convf5_d2_kernel(20736);
    znn::phi::hbw_array<float> tensor_convf5_d2_bias(48);
    znn::phi::hbw_array<float> tensor_conv5_d2_kernel(6912);
    znn::phi::hbw_array<float> tensor_conv5_d2_bias(48);
    znn::phi::hbw_array<float> tensor_convf6_d2_kernel(20736);
    znn::phi::hbw_array<float> tensor_convf6_d2_bias(48);
    znn::phi::hbw_array<float> tensor_conv6_d2_kernel(6912);
    znn::phi::hbw_array<float> tensor_conv6_d2_bias(48);
    znn::phi::hbw_array<float> tensor_conv4_d1_kernel(20736);
    znn::phi::hbw_array<float> tensor_conv4_d1_bias(48);
    znn::phi::hbw_array<float> tensor_conv5_d1_kernel(20736);
    znn::phi::hbw_array<float> tensor_conv5_d1_bias(48);
    znn::phi::hbw_array<float> tensor_conv6_d1_kernel(20736);
    znn::phi::hbw_array<float> tensor_conv6_d1_bias(48);
    znn::phi::hbw_array<float> tensor_conv4_d0_kernel(13824);
    znn::phi::hbw_array<float> tensor_conv4_d0_bias(32);
    znn::phi::hbw_array<float> tensor_conv5_d0_kernel(9216);
    znn::phi::hbw_array<float> tensor_conv5_d0_bias(32);
    znn::phi::hbw_array<float> tensor_conv6_d0_kernel(9216);
    znn::phi::hbw_array<float> tensor_conv6_d0_bias(32);
    znn::phi::hbw_array<float> tensor_conv7_d0_kernel(9216);
    znn::phi::hbw_array<float> tensor_conv7_d0_bias(32);

   std::vector<znn::phi::ConvWrapper> znnphi_convs(49);

   int conv_params[49][9];
   conv_params[0][0] = 1;
   conv_params[0][1] = 16;
   conv_params[0][2] = 32;
   conv_params[0][3] = 18;
   conv_params[0][4] = 192;
   conv_params[0][5] = 1;
   conv_params[0][6] = 5;
   conv_params[0][7] = 0;
   conv_params[0][8] = 2;
   conv_params[1][0] = 1;
   conv_params[1][1] = 32;
   conv_params[1][2] = 32;
   conv_params[1][3] = 18;
   conv_params[1][4] = 192;
   conv_params[1][5] = 1;
   conv_params[1][6] = 3;
   conv_params[1][7] = 0;
   conv_params[1][8] = 1;
   conv_params[2][0] = 1;
   conv_params[2][1] = 32;
   conv_params[2][2] = 32;
   conv_params[2][3] = 18;
   conv_params[2][4] = 192;
   conv_params[2][5] = 1;
   conv_params[2][6] = 3;
   conv_params[2][7] = 0;
   conv_params[2][8] = 1;
   conv_params[3][0] = 1;
   conv_params[3][1] = 32;
   conv_params[3][2] = 32;
   conv_params[3][3] = 18;
   conv_params[3][4] = 192;
   conv_params[3][5] = 1;
   conv_params[3][6] = 3;
   conv_params[3][7] = 0;
   conv_params[3][8] = 1;
   conv_params[4][0] = 1;
   conv_params[4][1] = 32;
   conv_params[4][2] = 48;
   conv_params[4][3] = 9;
   conv_params[4][4] = 96;
   conv_params[4][5] = 1;
   conv_params[4][6] = 3;
   conv_params[4][7] = 0;
   conv_params[4][8] = 1;
   conv_params[5][0] = 1;
   conv_params[5][1] = 48;
   conv_params[5][2] = 48;
   conv_params[5][3] = 9;
   conv_params[5][4] = 96;
   conv_params[5][5] = 1;
   conv_params[5][6] = 3;
   conv_params[5][7] = 0;
   conv_params[5][8] = 1;
   conv_params[6][0] = 1;
   conv_params[6][1] = 48;
   conv_params[6][2] = 48;
   conv_params[6][3] = 9;
   conv_params[6][4] = 96;
   conv_params[6][5] = 1;
   conv_params[6][6] = 3;
   conv_params[6][7] = 0;
   conv_params[6][8] = 1;
   conv_params[7][0] = 1;
   conv_params[7][1] = 48;
   conv_params[7][2] = 48;
   conv_params[7][3] = 5;
   conv_params[7][4] = 48;
   conv_params[7][5] = 1;
   conv_params[7][6] = 3;
   conv_params[7][7] = 0;
   conv_params[7][8] = 1;
   conv_params[8][0] = 1;
   conv_params[8][1] = 48;
   conv_params[8][2] = 48;
   conv_params[8][3] = 5;
   conv_params[8][4] = 48;
   conv_params[8][5] = 1;
   conv_params[8][6] = 3;
   conv_params[8][7] = 0;
   conv_params[8][8] = 1;
   conv_params[9][0] = 1;
   conv_params[9][1] = 48;
   conv_params[9][2] = 48;
   conv_params[9][3] = 5;
   conv_params[9][4] = 48;
   conv_params[9][5] = 3;
   conv_params[9][6] = 1;
   conv_params[9][7] = 1;
   conv_params[9][8] = 0;
   conv_params[10][0] = 1;
   conv_params[10][1] = 48;
   conv_params[10][2] = 48;
   conv_params[10][3] = 5;
   conv_params[10][4] = 48;
   conv_params[10][5] = 1;
   conv_params[10][6] = 3;
   conv_params[10][7] = 0;
   conv_params[10][8] = 1;
   conv_params[11][0] = 1;
   conv_params[11][1] = 48;
   conv_params[11][2] = 48;
   conv_params[11][3] = 5;
   conv_params[11][4] = 48;
   conv_params[11][5] = 3;
   conv_params[11][6] = 1;
   conv_params[11][7] = 1;
   conv_params[11][8] = 0;
   conv_params[12][0] = 1;
   conv_params[12][1] = 48;
   conv_params[12][2] = 64;
   conv_params[12][3] = 3;
   conv_params[12][4] = 24;
   conv_params[12][5] = 1;
   conv_params[12][6] = 3;
   conv_params[12][7] = 0;
   conv_params[12][8] = 1;
   conv_params[13][0] = 1;
   conv_params[13][1] = 64;
   conv_params[13][2] = 64;
   conv_params[13][3] = 3;
   conv_params[13][4] = 24;
   conv_params[13][5] = 1;
   conv_params[13][6] = 3;
   conv_params[13][7] = 0;
   conv_params[13][8] = 1;
   conv_params[14][0] = 1;
   conv_params[14][1] = 64;
   conv_params[14][2] = 64;
   conv_params[14][3] = 3;
   conv_params[14][4] = 24;
   conv_params[14][5] = 3;
   conv_params[14][6] = 1;
   conv_params[14][7] = 1;
   conv_params[14][8] = 0;
   conv_params[15][0] = 1;
   conv_params[15][1] = 64;
   conv_params[15][2] = 64;
   conv_params[15][3] = 3;
   conv_params[15][4] = 24;
   conv_params[15][5] = 1;
   conv_params[15][6] = 3;
   conv_params[15][7] = 0;
   conv_params[15][8] = 1;
   conv_params[16][0] = 1;
   conv_params[16][1] = 64;
   conv_params[16][2] = 64;
   conv_params[16][3] = 3;
   conv_params[16][4] = 24;
   conv_params[16][5] = 3;
   conv_params[16][6] = 1;
   conv_params[16][7] = 1;
   conv_params[16][8] = 0;
   conv_params[17][0] = 1;
   conv_params[17][1] = 64;
   conv_params[17][2] = 80;
   conv_params[17][3] = 2;
   conv_params[17][4] = 12;
   conv_params[17][5] = 1;
   conv_params[17][6] = 3;
   conv_params[17][7] = 0;
   conv_params[17][8] = 1;
   conv_params[18][0] = 1;
   conv_params[18][1] = 80;
   conv_params[18][2] = 80;
   conv_params[18][3] = 2;
   conv_params[18][4] = 12;
   conv_params[18][5] = 1;
   conv_params[18][6] = 3;
   conv_params[18][7] = 0;
   conv_params[18][8] = 1;
   conv_params[19][0] = 1;
   conv_params[19][1] = 80;
   conv_params[19][2] = 80;
   conv_params[19][3] = 2;
   conv_params[19][4] = 12;
   conv_params[19][5] = 3;
   conv_params[19][6] = 1;
   conv_params[19][7] = 1;
   conv_params[19][8] = 0;
   conv_params[20][0] = 1;
   conv_params[20][1] = 80;
   conv_params[20][2] = 80;
   conv_params[20][3] = 2;
   conv_params[20][4] = 12;
   conv_params[20][5] = 1;
   conv_params[20][6] = 3;
   conv_params[20][7] = 0;
   conv_params[20][8] = 1;
   conv_params[21][0] = 1;
   conv_params[21][1] = 80;
   conv_params[21][2] = 80;
   conv_params[21][3] = 2;
   conv_params[21][4] = 12;
   conv_params[21][5] = 3;
   conv_params[21][6] = 1;
   conv_params[21][7] = 1;
   conv_params[21][8] = 0;
   conv_params[22][0] = 1;
   conv_params[22][1] = 80;
   conv_params[22][2] = 96;
   conv_params[22][3] = 1;
   conv_params[22][4] = 6;
   conv_params[22][5] = 1;
   conv_params[22][6] = 3;
   conv_params[22][7] = 0;
   conv_params[22][8] = 1;
   conv_params[23][0] = 1;
   conv_params[23][1] = 96;
   conv_params[23][2] = 96;
   conv_params[23][3] = 1;
   conv_params[23][4] = 6;
   conv_params[23][5] = 1;
   conv_params[23][6] = 3;
   conv_params[23][7] = 0;
   conv_params[23][8] = 1;
   conv_params[24][0] = 1;
   conv_params[24][1] = 96;
   conv_params[24][2] = 96;
   conv_params[24][3] = 1;
   conv_params[24][4] = 6;
   conv_params[24][5] = 3;
   conv_params[24][6] = 1;
   conv_params[24][7] = 1;
   conv_params[24][8] = 0;
   conv_params[25][0] = 1;
   conv_params[25][1] = 96;
   conv_params[25][2] = 96;
   conv_params[25][3] = 1;
   conv_params[25][4] = 6;
   conv_params[25][5] = 1;
   conv_params[25][6] = 3;
   conv_params[25][7] = 0;
   conv_params[25][8] = 1;
   conv_params[26][0] = 1;
   conv_params[26][1] = 96;
   conv_params[26][2] = 96;
   conv_params[26][3] = 1;
   conv_params[26][4] = 6;
   conv_params[26][5] = 3;
   conv_params[26][6] = 1;
   conv_params[26][7] = 1;
   conv_params[26][8] = 0;
   conv_params[27][0] = 1;
   conv_params[27][1] = 96;
   conv_params[27][2] = 80;
   conv_params[27][3] = 1;
   conv_params[27][4] = 12;
   conv_params[27][5] = 1;
   conv_params[27][6] = 3;
   conv_params[27][7] = 0;
   conv_params[27][8] = 1;
   conv_params[28][0] = 1;
   conv_params[28][1] = 80;
   conv_params[28][2] = 80;
   conv_params[28][3] = 1;
   conv_params[28][4] = 12;
   conv_params[28][5] = 1;
   conv_params[28][6] = 3;
   conv_params[28][7] = 0;
   conv_params[28][8] = 1;
   conv_params[29][0] = 1;
   conv_params[29][1] = 80;
   conv_params[29][2] = 80;
   conv_params[29][3] = 1;
   conv_params[29][4] = 12;
   conv_params[29][5] = 3;
   conv_params[29][6] = 1;
   conv_params[29][7] = 1;
   conv_params[29][8] = 0;
   conv_params[30][0] = 1;
   conv_params[30][1] = 80;
   conv_params[30][2] = 80;
   conv_params[30][3] = 1;
   conv_params[30][4] = 12;
   conv_params[30][5] = 1;
   conv_params[30][6] = 3;
   conv_params[30][7] = 0;
   conv_params[30][8] = 1;
   conv_params[31][0] = 1;
   conv_params[31][1] = 80;
   conv_params[31][2] = 80;
   conv_params[31][3] = 1;
   conv_params[31][4] = 12;
   conv_params[31][5] = 3;
   conv_params[31][6] = 1;
   conv_params[31][7] = 1;
   conv_params[31][8] = 0;
   conv_params[32][0] = 1;
   conv_params[32][1] = 80;
   conv_params[32][2] = 64;
   conv_params[32][3] = 1;
   conv_params[32][4] = 24;
   conv_params[32][5] = 1;
   conv_params[32][6] = 3;
   conv_params[32][7] = 0;
   conv_params[32][8] = 1;
   conv_params[33][0] = 1;
   conv_params[33][1] = 64;
   conv_params[33][2] = 64;
   conv_params[33][3] = 1;
   conv_params[33][4] = 24;
   conv_params[33][5] = 1;
   conv_params[33][6] = 3;
   conv_params[33][7] = 0;
   conv_params[33][8] = 1;
   conv_params[34][0] = 1;
   conv_params[34][1] = 64;
   conv_params[34][2] = 64;
   conv_params[34][3] = 1;
   conv_params[34][4] = 24;
   conv_params[34][5] = 3;
   conv_params[34][6] = 1;
   conv_params[34][7] = 1;
   conv_params[34][8] = 0;
   conv_params[35][0] = 1;
   conv_params[35][1] = 64;
   conv_params[35][2] = 64;
   conv_params[35][3] = 1;
   conv_params[35][4] = 24;
   conv_params[35][5] = 1;
   conv_params[35][6] = 3;
   conv_params[35][7] = 0;
   conv_params[35][8] = 1;
   conv_params[36][0] = 1;
   conv_params[36][1] = 64;
   conv_params[36][2] = 64;
   conv_params[36][3] = 1;
   conv_params[36][4] = 24;
   conv_params[36][5] = 3;
   conv_params[36][6] = 1;
   conv_params[36][7] = 1;
   conv_params[36][8] = 0;
   conv_params[37][0] = 1;
   conv_params[37][1] = 64;
   conv_params[37][2] = 48;
   conv_params[37][3] = 1;
   conv_params[37][4] = 48;
   conv_params[37][5] = 1;
   conv_params[37][6] = 3;
   conv_params[37][7] = 0;
   conv_params[37][8] = 1;
   conv_params[38][0] = 1;
   conv_params[38][1] = 48;
   conv_params[38][2] = 48;
   conv_params[38][3] = 1;
   conv_params[38][4] = 48;
   conv_params[38][5] = 1;
   conv_params[38][6] = 3;
   conv_params[38][7] = 0;
   conv_params[38][8] = 1;
   conv_params[39][0] = 1;
   conv_params[39][1] = 48;
   conv_params[39][2] = 48;
   conv_params[39][3] = 1;
   conv_params[39][4] = 48;
   conv_params[39][5] = 3;
   conv_params[39][6] = 1;
   conv_params[39][7] = 1;
   conv_params[39][8] = 0;
   conv_params[40][0] = 1;
   conv_params[40][1] = 48;
   conv_params[40][2] = 48;
   conv_params[40][3] = 1;
   conv_params[40][4] = 48;
   conv_params[40][5] = 1;
   conv_params[40][6] = 3;
   conv_params[40][7] = 0;
   conv_params[40][8] = 1;
   conv_params[41][0] = 1;
   conv_params[41][1] = 48;
   conv_params[41][2] = 48;
   conv_params[41][3] = 1;
   conv_params[41][4] = 48;
   conv_params[41][5] = 3;
   conv_params[41][6] = 1;
   conv_params[41][7] = 1;
   conv_params[41][8] = 0;
   conv_params[42][0] = 1;
   conv_params[42][1] = 48;
   conv_params[42][2] = 48;
   conv_params[42][3] = 1;
   conv_params[42][4] = 96;
   conv_params[42][5] = 1;
   conv_params[42][6] = 3;
   conv_params[42][7] = 0;
   conv_params[42][8] = 1;
   conv_params[43][0] = 1;
   conv_params[43][1] = 48;
   conv_params[43][2] = 48;
   conv_params[43][3] = 1;
   conv_params[43][4] = 96;
   conv_params[43][5] = 1;
   conv_params[43][6] = 3;
   conv_params[43][7] = 0;
   conv_params[43][8] = 1;
   conv_params[44][0] = 1;
   conv_params[44][1] = 48;
   conv_params[44][2] = 48;
   conv_params[44][3] = 1;
   conv_params[44][4] = 96;
   conv_params[44][5] = 1;
   conv_params[44][6] = 3;
   conv_params[44][7] = 0;
   conv_params[44][8] = 1;
   conv_params[45][0] = 1;
   conv_params[45][1] = 48;
   conv_params[45][2] = 32;
   conv_params[45][3] = 1;
   conv_params[45][4] = 192;
   conv_params[45][5] = 1;
   conv_params[45][6] = 3;
   conv_params[45][7] = 0;
   conv_params[45][8] = 1;
   conv_params[46][0] = 1;
   conv_params[46][1] = 32;
   conv_params[46][2] = 32;
   conv_params[46][3] = 1;
   conv_params[46][4] = 192;
   conv_params[46][5] = 1;
   conv_params[46][6] = 3;
   conv_params[46][7] = 0;
   conv_params[46][8] = 1;
   conv_params[47][0] = 1;
   conv_params[47][1] = 32;
   conv_params[47][2] = 32;
   conv_params[47][3] = 1;
   conv_params[47][4] = 192;
   conv_params[47][5] = 1;
   conv_params[47][6] = 3;
   conv_params[47][7] = 0;
   conv_params[47][8] = 1;
   conv_params[48][0] = 1;
   conv_params[48][1] = 32;
   conv_params[48][2] = 32;
   conv_params[48][3] = 1;
   conv_params[48][4] = 192;
   conv_params[48][5] = 1;
   conv_params[48][6] = 3;
   conv_params[48][7] = 0;
   conv_params[48][8] = 1;

//   omp_set_dynamic(20);     // Explicitly disable dynamic teams
//   omp_set_num_threads(20); // Use 4 threads for all consecutive parallel regions
//#pragma omp parallel for
   for (int i = 0; i < 49; i++) {
      znnphi_convs[i].init(conv_params[i][0], conv_params[i][1], conv_params[i][2],
                           conv_params[i][3], conv_params[i][4], conv_params[i][5],
                           conv_params[i][6], conv_params[i][7], conv_params[i][8]);
   } // for (0..49)
   std::cout << "Starting the run...\n";
{
    auto begin = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 1; i++) {
    znnphi_convs[0].forward(tensor_input.data(), 
                           tensor_convi.data(), 
                           tensor_convi_kernel.data(), 
                           tensor_convi_bias.data());
    znnphi_convs[1].forward(tensor_convi.data(), 
                           tensor_conv0_d0.data(), 
                           tensor_conv0_d0_kernel.data(), 
                           tensor_conv0_d0_bias.data());
    znnphi_convs[2].forward(tensor_conv0_d0.data(), 
                           tensor_conv1_d0.data(), 
                           tensor_conv1_d0_kernel.data(), 
                           tensor_conv1_d0_bias.data());
    znnphi_convs[3].forward(tensor_conv1_d0.data(), 
                           tensor_conv2_d0.data(), 
                           tensor_conv2_d0_kernel.data(), 
                           tensor_conv2_d0_bias.data());
    znnphi_convs[4].forward(tensor_pool_d1.data(), 
                           tensor_conv0_d1.data(), 
                           tensor_conv0_d1_kernel.data(), 
                           tensor_conv0_d1_bias.data());
    znnphi_convs[5].forward(tensor_conv0_d1.data(), 
                           tensor_conv1_d1.data(), 
                           tensor_conv1_d1_kernel.data(), 
                           tensor_conv1_d1_bias.data());
    znnphi_convs[6].forward(tensor_conv1_d1.data(), 
                           tensor_conv2_d1.data(), 
                           tensor_conv2_d1_kernel.data(), 
                           tensor_conv2_d1_bias.data());
    znnphi_convs[7].forward(tensor_pool_d2.data(), 
                           tensor_conv0_d2.data(), 
                           tensor_conv0_d2_kernel.data(), 
                           tensor_conv0_d2_bias.data());
    znnphi_convs[8].forward(tensor_conv0_d2.data(), 
                           tensor_convf1_d2.data(), 
                           tensor_convf1_d2_kernel.data(), 
                           tensor_convf1_d2_bias.data());
    znnphi_convs[9].forward(tensor_convf1_d2.data(), 
                           tensor_conv1_d2.data(), 
                           tensor_conv1_d2_kernel.data(), 
                           tensor_conv1_d2_bias.data());
    znnphi_convs[10].forward(tensor_conv1_d2.data(), 
                           tensor_convf2_d2.data(), 
                           tensor_convf2_d2_kernel.data(), 
                           tensor_convf2_d2_bias.data());
    znnphi_convs[11].forward(tensor_convf2_d2.data(), 
                           tensor_conv2_d2.data(), 
                           tensor_conv2_d2_kernel.data(), 
                           tensor_conv2_d2_bias.data());
    znnphi_convs[12].forward(tensor_pool_d3.data(), 
                           tensor_conv0_d3.data(), 
                           tensor_conv0_d3_kernel.data(), 
                           tensor_conv0_d3_bias.data());
    znnphi_convs[13].forward(tensor_conv0_d3.data(), 
                           tensor_convf1_d3.data(), 
                           tensor_convf1_d3_kernel.data(), 
                           tensor_convf1_d3_bias.data());
    znnphi_convs[14].forward(tensor_convf1_d3.data(), 
                           tensor_conv1_d3.data(), 
                           tensor_conv1_d3_kernel.data(), 
                           tensor_conv1_d3_bias.data());
    znnphi_convs[15].forward(tensor_conv1_d3.data(), 
                           tensor_convf2_d3.data(), 
                           tensor_convf2_d3_kernel.data(), 
                           tensor_convf2_d3_bias.data());
    znnphi_convs[16].forward(tensor_convf2_d3.data(), 
                           tensor_conv2_d3.data(), 
                           tensor_conv2_d3_kernel.data(), 
                           tensor_conv2_d3_bias.data());
    znnphi_convs[17].forward(tensor_pool_d4.data(), 
                           tensor_conv0_d4.data(), 
                           tensor_conv0_d4_kernel.data(), 
                           tensor_conv0_d4_bias.data());
    znnphi_convs[18].forward(tensor_conv0_d4.data(), 
                           tensor_convf1_d4.data(), 
                           tensor_convf1_d4_kernel.data(), 
                           tensor_convf1_d4_bias.data());
    znnphi_convs[19].forward(tensor_convf1_d4.data(), 
                           tensor_conv1_d4.data(), 
                           tensor_conv1_d4_kernel.data(), 
                           tensor_conv1_d4_bias.data());
    znnphi_convs[20].forward(tensor_conv1_d4.data(), 
                           tensor_convf2_d4.data(), 
                           tensor_convf2_d4_kernel.data(), 
                           tensor_convf2_d4_bias.data());
    znnphi_convs[21].forward(tensor_convf2_d4.data(), 
                           tensor_conv2_d4.data(), 
                           tensor_conv2_d4_kernel.data(), 
                           tensor_conv2_d4_bias.data());
    znnphi_convs[22].forward(tensor_pool_d5.data(), 
                           tensor_conv0_d5.data(), 
                           tensor_conv0_d5_kernel.data(), 
                           tensor_conv0_d5_bias.data());
    znnphi_convs[23].forward(tensor_conv0_d5.data(), 
                           tensor_convf1_d5.data(), 
                           tensor_convf1_d5_kernel.data(), 
                           tensor_convf1_d5_bias.data());
    znnphi_convs[24].forward(tensor_convf1_d5.data(), 
                           tensor_conv1_d5.data(), 
                           tensor_conv1_d5_kernel.data(), 
                           tensor_conv1_d5_bias.data());
    znnphi_convs[25].forward(tensor_conv1_d5.data(), 
                           tensor_convf2_d5.data(), 
                           tensor_convf2_d5_kernel.data(), 
                           tensor_convf2_d5_bias.data());
    znnphi_convs[26].forward(tensor_convf2_d5.data(), 
                           tensor_conv2_d5.data(), 
                           tensor_conv2_d5_kernel.data(), 
                           tensor_conv2_d5_bias.data());
    znnphi_convs[27].forward(tensor_Eltwise1.data(), 
                           tensor_conv4_d4.data(), 
                           tensor_conv4_d4_kernel.data(), 
                           tensor_conv4_d4_bias.data());
    znnphi_convs[28].forward(tensor_conv4_d4.data(), 
                           tensor_convf5_d4.data(), 
                           tensor_convf5_d4_kernel.data(), 
                           tensor_convf5_d4_bias.data());
    znnphi_convs[29].forward(tensor_convf5_d4.data(), 
                           tensor_conv5_d4.data(), 
                           tensor_conv5_d4_kernel.data(), 
                           tensor_conv5_d4_bias.data());
    znnphi_convs[30].forward(tensor_conv5_d4.data(), 
                           tensor_convf6_d4.data(), 
                           tensor_convf6_d4_kernel.data(), 
                           tensor_convf6_d4_bias.data());
    znnphi_convs[31].forward(tensor_convf6_d4.data(), 
                           tensor_conv6_d4.data(), 
                           tensor_conv6_d4_kernel.data(), 
                           tensor_conv6_d4_bias.data());
    znnphi_convs[32].forward(tensor_Eltwise2.data(), 
                           tensor_conv4_d3.data(), 
                           tensor_conv4_d3_kernel.data(), 
                           tensor_conv4_d3_bias.data());
    znnphi_convs[33].forward(tensor_conv4_d3.data(), 
                           tensor_convf5_d3.data(), 
                           tensor_convf5_d3_kernel.data(), 
                           tensor_convf5_d3_bias.data());
    znnphi_convs[34].forward(tensor_convf5_d3.data(), 
                           tensor_conv5_d3.data(), 
                           tensor_conv5_d3_kernel.data(), 
                           tensor_conv5_d3_bias.data());
    znnphi_convs[35].forward(tensor_conv5_d3.data(), 
                           tensor_convf6_d3.data(), 
                           tensor_convf6_d3_kernel.data(), 
                           tensor_convf6_d3_bias.data());
    znnphi_convs[36].forward(tensor_convf6_d3.data(), 
                           tensor_conv6_d3.data(), 
                           tensor_conv6_d3_kernel.data(), 
                           tensor_conv6_d3_bias.data());
    znnphi_convs[37].forward(tensor_Eltwise3.data(), 
                           tensor_conv4_d2.data(), 
                           tensor_conv4_d2_kernel.data(), 
                           tensor_conv4_d2_bias.data());
    znnphi_convs[38].forward(tensor_conv4_d2.data(), 
                           tensor_convf5_d2.data(), 
                           tensor_convf5_d2_kernel.data(), 
                           tensor_convf5_d2_bias.data());
    znnphi_convs[39].forward(tensor_convf5_d2.data(), 
                           tensor_conv5_d2.data(), 
                           tensor_conv5_d2_kernel.data(), 
                           tensor_conv5_d2_bias.data());
    znnphi_convs[40].forward(tensor_conv5_d2.data(), 
                           tensor_convf6_d2.data(), 
                           tensor_convf6_d2_kernel.data(), 
                           tensor_convf6_d2_bias.data());
    znnphi_convs[41].forward(tensor_convf6_d2.data(), 
                           tensor_conv6_d2.data(), 
                           tensor_conv6_d2_kernel.data(), 
                           tensor_conv6_d2_bias.data());
    znnphi_convs[42].forward(tensor_Eltwise4.data(), 
                           tensor_conv4_d1.data(), 
                           tensor_conv4_d1_kernel.data(), 
                           tensor_conv4_d1_bias.data());
    znnphi_convs[43].forward(tensor_conv4_d1.data(), 
                           tensor_conv5_d1.data(), 
                           tensor_conv5_d1_kernel.data(), 
                           tensor_conv5_d1_bias.data());
    znnphi_convs[44].forward(tensor_conv5_d1.data(), 
                           tensor_conv6_d1.data(), 
                           tensor_conv6_d1_kernel.data(), 
                           tensor_conv6_d1_bias.data());
    znnphi_convs[45].forward(tensor_merge_d3.data(), 
                           tensor_conv4_d0.data(), 
                           tensor_conv4_d0_kernel.data(), 
                           tensor_conv4_d0_bias.data());
    znnphi_convs[46].forward(tensor_conv4_d0.data(), 
                           tensor_conv5_d0.data(), 
                           tensor_conv5_d0_kernel.data(), 
                           tensor_conv5_d0_bias.data());
    znnphi_convs[47].forward(tensor_conv5_d0.data(), 
                           tensor_conv6_d0.data(), 
                           tensor_conv6_d0_kernel.data(), 
                           tensor_conv6_d0_bias.data());
    znnphi_convs[48].forward(tensor_sum4_d0.data(), 
                           tensor_conv7_d0.data(), 
                           tensor_conv7_d0_kernel.data(), 
                           tensor_conv7_d0_bias.data());
}
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
            (end - begin).count();
    double secs = static_cast<double>(duration) / 1000000;
    std::cout << "total: " << secs/1 << "\n";
}

} //int main
