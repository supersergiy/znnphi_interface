#!/bin/bash
for run in {1..30}
do
   $1 &> tmp.txt
   echo "Nan is there: "
   grep -c nan tmp.txt
done
