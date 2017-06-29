#!/bin/bash
make bin N=./reference/unet/unet.json W=./reference/unet/unet.h5; out/znet.bin > tmp.txt

