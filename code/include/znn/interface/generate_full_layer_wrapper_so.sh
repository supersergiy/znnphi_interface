#!/bin/bash
BASE_PATH=$ZNNPHI_PATH/include/znn/interface

USER_PROVIDED_DL_PATH=$1
BN=$2
IFM=$3
OFM=$4
ID=$5
IHW=$6
KD=$7
KHW=$8
PADD=$9
PADHW=$10
CORES=$11
HT=$12

POSTFIX=${BN}_${IFM}_${OFM}_${ID}_${IHW}_${KD}_${KHW}_${PADD}_${PADHW}_${CORES}_${HT}

DL_DIR=$BASE_PATH/dl_$POSTFIX
SO_NAME=conv_wrapper_${POSTFIX}.so
SO_PATH=$SO_DIR/$SO_NAME
PARAMS_FILE_NAME=params.hpp

createParamsFile () 
{ 
   if ! [ -e "$PARAMS_FILE_NAME" ]
   then
      rm "$PARAMS_FILE_NAME"
   fi
   touch "$PARAMS_FILE_NAME"
   echo "#pragma once" >> "$PARAMS_FILE_NAME"

   echo "#define BN_v    $BN"    >> "$PARAMS_FILE_NAME"
   echo "#define IFM_v   $IFM"   >> "$PARAMS_FILE_NAME"
   echo "#define OFM_v   $OFM"   >> "$PARAMS_FILE_NAME"
   echo "#define ID_v    $ID"    >> "$PARAMS_FILE_NAME"
   echo "#define IHW_v   $IHW"   >> "$PARAMS_FILE_NAME"
   echo "#define KD_v    $KD"    >> "$PARAMS_FILE_NAME"
   echo "#define KHW_v   $KHW"   >> "$PARAMS_FILE_NAME"
   echo "#define PADD_v  $PADD"  >> "$PARAMS_FILE_NAME"
   echo "#define PADHW_v $PADHW" >> "$PARAMS_FILE_NAME"
   echo "#define CORES_v $CORES" >> "$PARAMS_FILE_NAME"
   echo "#define HT_v    $HT"    >> "$PARAMS_FILE_NAME"
}

# If the Shared Object that user wants is already there, 
# no need to do anything
if ! [ -e "$USER_PROVIDED_SO_PATH" ]
then
   # Recompile if a shared object with the same params in not already there 
   if ! [ -e "$SO_PATH" ]
   then
      echo "Compiling specialized full layer..."
      if ! [ -e "$SO_DIR" ]
      then 
          mkdir "$SO_DIR"
      fi

      cp ZnnPhiConvWrapper.cpp ZnnPhiConvWrapper.hpp "$SO_DIR"
      cd "$SO_DIR"
      createParamsFile

      #SSE_FLAGS=-DZNN_SSE -msse 
      AVX_FLAGS=-DZNN_AVX -xAVX
      #AVX2_FLAGS=-DZNN_AVX2 -march=core-avx2 
      #AVX512_FLAGS=-DZNN_AVX512 -xMIC-AVX512 
      #KNC_FLAGS=-DZNN_KNC -mmic 

      STANDARD_FLAG=-std=c++14
      CXX=g++-5 -pthread $(STANDARD_FLAG)
      OPTIMIZATION_FLAGS=-DNDEBUG -O3 -ffast-math -fno-omit-frame-pointer -fno-rtti -fno-exceptions
      CS_LD_FLAGS=-lpthread
      FPIc=-fPIC

      CXXINCLUDES=-I$(HERE)/include -I$(HERE)/..
      CXXWARN=-Wall -Wextra -Wno-format -Wno-unknown-pragmas
      HBW_FLAG=-DZNN_NO_HBW      
      g++ -fPIC -shared ZnnPhiConvWrapper.cpp -o "$SO_PATH" 
      cd ..
   fi

   echo "Copying $SO_PATH into $USER_PROVIDED_SO_PATH..."
   cp "$SO_PATH" "$USER_PROVIDED_SO_PATH"
else
   echo "Shared object $USER_PROVIDED_SO_PATH aready exists."
fi


