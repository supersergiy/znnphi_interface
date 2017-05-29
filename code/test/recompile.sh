#!/bin/bash
make bin/avx2/test_un_block.bin ICC=1 OPT=1 NOHBW=1 &> compile.txt
