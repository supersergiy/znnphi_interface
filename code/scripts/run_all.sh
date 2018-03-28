#!/bin/bash
#git checkout benchmarking
#./run_single_test.py /tests/unet_paper full_opt
#./run_single_test.py /tests/unet_paper_bn full_opt
./run_single_test.py /tests/unet_paper_elu full_opt
./run_single_test.py /tests/unet_paper_bn_elu full_opt

./run_single_test.py /tests/unet_paper_bn no_opt
./run_single_test.py /tests/unet_paper_bn no_add
./run_single_test.py /tests/unet_paper_bn no_act
./run_single_test.py /tests/unet_paper_bn no_pad

./run_single_test.py /tests/unet_paper_bn_elu no_opt
./run_single_test.py /tests/unet_paper_bn_elu no_add
./run_single_test.py /tests/unet_paper_bn_elu no_act
./run_single_test.py /tests/unet_paper_bn_elu no_pad

./run_single_test.py /tests/unet_sym full_opt
./run_single_test.py /tests/unet_sym_bn full_opt
./run_single_test.py /tests/unet_sym_elu full_opt
./run_single_test.py /tests/unet_sym_bn_elu full_opt

./run_single_test.py /tests/unet_sym_bn no_opt
./run_single_test.py /tests/unet_sym_bn no_add
./run_single_test.py /tests/unet_sym_bn no_act
./run_single_test.py /tests/unet_sym_bn no_pad

./run_single_test.py /tests/unet_sym_bn_elu no_opt
./run_single_test.py /tests/unet_sym_bn_elu no_add
./run_single_test.py /tests/unet_sym_bn_elu no_act
./run_single_test.py /tests/unet_sym_bn_elu no_pad

git checkout benchmarking_pni
#./run_single_test.py /tests/pni_unet full_opt
./run_single_test.py /tests/pni_unet_no_bn full_opt
./run_single_test.py /tests/pni_unet_relu full_opt
./run_single_test.py /tests/pni_unet_no_bn_relu full_opt

#./run_single_test.py /tests/pni_unet no_opt
./run_single_test.py /tests/pni_unet no_add
./run_single_test.py /tests/pni_unet no_act
./run_single_test.py /tests/pni_unet no_pad

./run_single_test.py /tests/pni_unet_relu no_opt
./run_single_test.py /tests/pni_unet_relu no_add
./run_single_test.py /tests/pni_unet_relu no_act
./run_single_test.py /tests/pni_unet_relu no_pad
git checkout benchmarking
