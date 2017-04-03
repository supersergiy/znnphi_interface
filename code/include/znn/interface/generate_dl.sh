#!/bin/bash
USER_PROVIDED_DL_PATH=$1
BN=$2
IFM=$3
OFM=$4
ID=$5
IHW=$6
KD=$7
KHW=$8
PADD=$9
PADHW=${10}
CORES=${11}
HT=${12}

POSTFIX=${BN}_${IFM}_${OFM}_${ID}_${IHW}_${KD}_${KHW}_${PADD}_${PADHW}_${CORES}_${HT}

BASE_PATH=$ZNNPHI_PATH/include/znn/interface
DL_FILES_DIR=${BASE_PATH}/dl_files
DL_DIR=${DL_FILES_DIR}/dl_$POSTFIX
DL_NAME=conv_wrapper_${POSTFIX}.so
DL_PATH=$DL_DIR/$DL_NAME
PARAMS_FILE_NAME=params.hpp

createParamsFile () 
{ 
   if [ -e "$PARAMS_FILE_NAME" ]
   then
      rm "$PARAMS_FILE_NAME"
   fi
   touch "$PARAMS_FILE_NAME"
   echo "#pragma once" >> "$PARAMS_FILE_NAME"

   echo "#define B_v     $BN"    >> "$PARAMS_FILE_NAME"
   echo "#define IFM_v   $IFM"   >> "$PARAMS_FILE_NAME"
   echo "#define OFM_v   $OFM"   >> "$PARAMS_FILE_NAME"
   echo "#define ID_v    $ID"    >> "$PARAMS_FILE_NAME"
   echo "#define IHW_v   $IHW"   >> "$PARAMS_FILE_NAME"
   echo "#define KD_v    $KD"    >> "$PARAMS_FILE_NAME"
   echo "#define KHW_v   $KHW"   >> "$PARAMS_FILE_NAME"
   echo "#define PADD_v  $PADD"  >> "$PARAMS_FILE_NAME"
   echo "#define PADHW_v $PADHW" >> "$PARAMS_FILE_NAME"
   echo "#define Cores_v $CORES" >> "$PARAMS_FILE_NAME"
   echo "#define HT_v    $HT"    >> "$PARAMS_FILE_NAME"
}

# If the Shared Object that user wants is already there, 
# no need to do anything
if ! [ -e "$USER_PROVIDED_DL_PATH" ]
then
   # Recompile if a shared object with the same params in not already there 
   if ! [ -e "$DL_PATH" ]
   then
      cd "$BASE_PATH"
      echo "Recompiling layer..."
      if ! [ -e "$DL_DIR" ]
      then 
          mkdir "$DL_DIR"
      fi
      cp ConvMakefile ZnnPhiConvWrapper.cpp ZnnPhiConvWrapper.hpp "$DL_DIR"
      cd "$DL_DIR"
      cp ConvMakefile Makefile
      createParamsFile
      make dl DL_NAME=$DL_NAME
      cd ..
   fi

   #echo "Copying $DL_PATH into $USER_PROVIDED_DL_PATH..."
   cp "$DL_PATH" "$USER_PROVIDED_DL_PATH"
else
   echo "Reusing layer"
fi


