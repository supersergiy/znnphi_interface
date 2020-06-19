HERE    = .
AT		=	@
DOLLAR  = 	$$

CD		=	$(AT)cd
CP		=	$(AT)cp
ECHO	=	@echo
CAT		=	$(AT)cat
IF		=	$(AT)if
LN		=	$(AT)ln
MKDIR	=	$(AT)mkdir
MV		=	$(AT)mv
SED		=	$(AT)sed
RM		=	$(AT)rm
TOUCH	=	$(AT)touch
CHMOD	=	$(AT)chmod
DATE    =   $(AT)date
BASE 	= ../../../..

# compilers

ZNN_TUNE		=
SSE_FLAGS   	=   -DZNN_SSE -msse $(ZNN_TUNE)                                               
AVX_FLAGS   	=   -DZNN_AVX -xAVX $(ZNN_TUNE)                                               
AVX2_FLAGS  	=   -DZNN_AVX2 -march=core-avx2 $(ZNN_TUNE)                                   
AVX512_FLAGS    =   -DZNN_AVX512 -xCORE-AVX512$(ZNN_TUNE)                                 
KNC_FLAGS   	=   -DZNN_KNC -mmic $(ZNN_TUNE)
CORES_FLAG  	=   -DZNN_NUM_CORES=2

STANDARD_FLAG	=	-std=c++14
STD_FLAG		=

CXX             =   $(AT)icpc $(STANDARD_FLAG)                                          

# add -DNDEBUG for debug mode
OPTIMIZATION_FLAGS  =   -DNDEBUG -O3 -ipo -fp-model fast=2 -mkl=sequential -fno-rtti -fno-exceptions 

CS_FLAGS        =   -mkl=sequential                                                       
CS_LD_FLAGS     =   -mkl=sequential  -static-intel 

FPIC		=	-fPIC

CXXINCLUDES	=	-I$(BASE)/cpp/include -I$(BASE)/cpp/include/znn 
CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas
CXXERR		=   -fmax-errors=5

HBW_LD_FLAG	=   #-lmemkind 
HBW_FLAG	=	-DZNN_NO_HBW


DEPFLAGS	=	-MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)" $(CS_FLAGS) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG)
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG) $(STD_FLAG)

COMMON_FLAGS	=	$(INLINE_DEPFLAGS) $(CXXINCLUDES) $(CXXWARN) $(CXXERR) 

DBG_FLAGS		=	$(COMMON_FLAGS) $(CS_FLAGS)

# add -DNDEBUG for debug
LD_OPTIMIZATION_FLAGS	=	-DNDEBUG -O3

COMMON_LDFLAGS		=	$(HBW_LD_FLAG)

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)
DBG_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) -g
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) $(LD_OPTIMIZATION_FLAGS)

#CXXFLAGS	=	$(DBG_FLAGS)
CXXFLAGS   	=	$(OPT_FLAGS)
CXXLDFLAGS	=	$(OPT_LDFLAGS)
DLFLAGS		= 	-fPIC -shared 


ifneq ("$(wildcard $(O))","")
	LAYER_EXISTS=1
else
	LAYER_EXISTS=0
endif

ifeq ($(strip $(ARCH)),AVX512)                                                                                                                                                                             
	ARCH_FLAGS=$(AVX512_FLAGS)                                                                                                                                                                             
else ifeq ($(strip $(ARCH)),AVX2)
	ARCH_FLAGS=$(AVX2_FLAGS) 
else                                                                                                                                                                                                       
$(error Unsupported or unspecified architecture: ARCH==$(ARCH). Please set Makefile ARCH parameter to one of [AVX2, AVX512].)                                                                              
endif     

#ZNNJIT_PATH = $(ZNNPHI_PATH)/cpp/include/znn/jit

FORCE:

/tmp/pznet_jit/%.so: /tmp/pznet_jit/%.cpp FORCE
ifeq ($(LAYER_EXISTS),0)
	#echo "\n$(CXX)         -c $(CXXFLAGS)  $(ARCH_FLAGS) $(PARAM_DEFS) -o $<.o $< \n"
	$(CXX)         -c $(CXXFLAGS)  $(ARCH_FLAGS) $(PARAM_DEFS) -o $<.o $<
	#echo "\n$(CXX)         $(DLFLAGS) $(CXXLDFLAGS) $(ARCH_FLAGS) -o $@ $<.o -ldl \n"
	$(CXX)         $(DLFLAGS) $(CXXLDFLAGS) $(ARCH_FLAGS) -o $@ $<.o -ldl
	$(RM)   $<.o
	$(MV)   $@ $(O)
else 
	$(ECHO) 
endif

all: clean test_jit.bin 

test_jit.bin: FORCE
	$(ECHO) $(CXX) $(CXXFLAGS) $(ARCH_FLAGS) ./test/test_jit.cpp -o $@ 
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) ./test/test_jit.cpp -o $@ 

client.bin: FORCE
	$(ECHO) $(CXX) $(CXXFLAGS) $(ARCH_FLAGS) ./client.cpp -o $@ 
	$(CXX) $(CXXFLAGS) $(ARCH2_FLAGS) ./client.cpp -o $@ 
clean:
	$(RM) -rf ./*.o ./*.so ./*.S *.bin /tmp/*.so /tmp/*.cpp 

remake: clean all


