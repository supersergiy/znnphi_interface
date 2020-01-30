#!/bin/bash
rm $1/*.h5
./generate_weights.py $1
./generate_reference.py $1
