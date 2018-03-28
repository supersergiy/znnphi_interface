#!/bin/bash
./run_single_test.py /tests/unet_paper full_opt
./run_single_test.py /tests/unet_paper_bn full_opt
git checkout bench_pni
./run_single_test.py /tests/pni_unet full_opt
./run_single_test.py /tests/pni_unet_relu full_opt
