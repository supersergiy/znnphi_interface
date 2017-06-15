#!/bin/bash
rm bin/avx2/${1}.bin
make ICC=1 OPT=1 CORES=2 NOHBW=1 -j2 bin/avx2/${1}.bin
./bin/avx2/${1}.bin
