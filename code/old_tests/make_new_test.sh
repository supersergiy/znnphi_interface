#!/bin/bash

name=$1
test_folder=new_tests/$name

rm -rf $test_folder
mkdir $test_folder 

./do_thing.sh

cp nets/net.prototxt $test_folder/net.prototxt
cp data/inputs/input.h5 $test_folder/in.h5
cp data/weights/weights.h5 $test_folder/weights.h5
cp data/reference/reference.h5 $test_folder/out.h5
