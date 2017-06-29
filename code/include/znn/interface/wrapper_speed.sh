#!/bin/bash
rm test_wrapper.bin
rm ~/znnphi_interface/lib/*
make test_wrapper.bin
time ./test_wrapper.bin
time ./test_wrapper.bin
