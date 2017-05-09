HERE    	=  ${ZNNPHI_PATH}/code/include/znn/interface
AT	=	@
DOLLAR  = 	$$

CD	=	$(AT)cd
CP	=	$(AT)cp
ECHO	=	@echo
CAT	=	$(AT)cat
IF	=	$(AT)if
LN	=	$(AT)ln
MKDIR	=	$(AT)mkdir
MV	=	$(AT)mv
SED	=	$(AT)sed
RM	=	$(AT)rm
TOUCH	=	$(AT)touch
CHMOD	=	$(AT)chmod
DATE    =       $(AT)date

# compilers

ZNN_TUNE	=
SSE_FLAGS   =   -DZNN_SSE -msse $(ZNN_TUNE)                                               
AVX_FLAGS   =   -DZNN_AVX -xAVX $(ZNN_TUNE)                                               
AVX2_FLAGS  =   -DZNN_AVX2 -march=core-avx2 $(ZNN_TUNE)                                   
AVX512_FLAGS    =   -DZNN_AVX512 -xMIC-AVX512 $(ZNN_TUNE)                                 
KNC_FLAGS   =   -DZNN_KNC -mmic $(ZNN_TUNE)
CORES_FLAG  =   -DZNN_NUM_CORES=64

STANDARD_FLAG	=	-std=c++14
STD_FLAG	=

CXX             =   $(AT)icpc $(STANDARD_FLAG)                                           
OPTIMIZATION_FLAGS  =   -DNDEBUG -O3 -ipo -fp-model fast=2 -mkl=sequential -fno-rtti -fno-exceptions 
CS_FLAGS        =   -mkl=sequential                                                       
CS_LD_FLAGS     =   -mkl=sequential  -static-intel 

FPIC	=	-fPIC

CXXINCLUDES	=	-I$(ZNNPHI_PATH)/code/include -I$(ZNNPHI_PATH)/code/include/znn -I$(ZNNPHI_PATH)
CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas
CXXERR		=   -fmax-errors=5

HBW_LD_FLAG	=   #-lmemkind 
HBW_FLAG	=	-DZNN_NO_HBW


DEPFLAGS	=	-MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)" $(CS_FLAGS) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG)
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG) $(STD_FLAG)

COMMON_FLAGS	=	$(INLINE_DEPFLAGS) $(CXXINCLUDES) $(CXXWARN) $(CXXERR) 

DBG_FLAGS		=	$(COMMON_FLAGS) $(CS_FLAGS)

LD_OPTIMIZATION_FLAGS	=	-DNDEBUG -O3
COMMON_LDFLAGS		=	$(HBW_LD_FLAG)

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)
DBG_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) -g
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) $(LD_OPTIMIZATION_FLAGS)

#CXXFLAGS	=	$(DBG_FLAGS)
CXXFLAGS   	=	$(OPT_FLAGS)
CXXLDFLAGS	=	$(OPT_LDFLAGS)
DLFLAGS		= 	-fPIC -shared 

# user gives these params to make layer
PARAM_DEFS  = -DBN_v=${BN} -DIFM_v=${IFM} -DOFM_v=${OFM} -DID_v=${ID} \
				  -DIHW_v=${IHW} -DKD_v=${KD} -DKHW_v=${KHW} -DPADD_v=${PADD} \
				  -DPADHW_v=${PADHW} -DCores_v=${CORES} -DHT_v=${HT}

# Note the trailing _ -- it has to be there
LAYER_NAME  =  conv_layer_${BN}_${IFM}_${OFM}_${ID}_${IHW}_${KD}_${KHW}_${PADD}_${PADHW}_${CORES}_${HT}_.so

LAYER_SO_PATH = ${ZNNPHI_PATH}/lib/${LAYER_NAME}

ifneq ("$(wildcard $(LAYER_SO_PATH))","")
	LAYER_EXISTS=1
else
	LAYER_EXISTS=0
endif

UNAME_S = $(shell uname -s)

ifeq ($(UNAME_S),Linux)
	LIB_DEST=/usr/lib
else
	LIB_DEST=/usr/lib64
endif

FORCE:

%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(CXX) $(DEPFLAGS) $(AVX512_FLAGS) -MF $@ $<remake

%.o: ./%.cpp FORCE
ifeq ($(LAYER_EXISTS),0)
	$(ECHO) "[CXX] compiling $<"
	$(ECHO) $(CXX) -c $(CXXFLAGS) $(AVX512_FLAGS) $(PARAM_DEFS) -o $@ $<
	$(CXX)         -c $(CXXFLAGS) $(AVX512_FLAGS) $(PARAM_DEFS) -o $@ $<
else
	$(ECHO) All good 
endif

%.so: ./%.o
ifeq ($(LAYER_EXISTS),0)
	$(ECHO) "[CXX] creating shared lib $@" 
	$(ECHO) $(CXX) $(DLFLAGS) $(CXXLDFLAGS) -o $@ $< 
	$(CXX)         $(DLFLAGS) $(CXXLDFLAGS) -o $@ $< 
else 
	$(ECHO) All good 
endif

.PHONY: wrapper, layer, all, clean, remake 
wrapper: conv_wrapper.so 
ifeq ($(LAYER_EXISTS),0)
	$(ECHO) ${UNAME_S}
	$(ECHO) ${LIB_DEST}
	sudo mv ${HERE}/$< ${LIB_DEST}/libznnphiconv.so
endif
layer: conv_layer.so
ifeq ($(LAYER_EXISTS),0)
	mv ${HERE}/$< ${LAYER_SO_PATH} 
endif
all: clean dl
clean:
	$(RM) -rf ./*.o ./*.so ./*.S *.bin 

remake: clean all
