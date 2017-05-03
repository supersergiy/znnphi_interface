#!/bin/bash
make ICC=1 OPT=1 CORES=64 NOHBW=1 -j64 bin/avx512/${1}.bin #; bin/avx512/${1}.bin 
