HERE    	= .
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
BASE = ../../../..

# compilers

ZNN_TUNE	=
SSE_FLAGS   =   -DZNN_SSE -msse $(ZNN_TUNE)                                               
AVX_FLAGS   =   -DZNN_AVX -xAVX $(ZNN_TUNE)                                               
AVX2_FLAGS  =   -DZNN_AVX2 -march=core-avx2 $(ZNN_TUNE)                                   
AVX512_FLAGS    =   -DZNN_AVX512 -xMIC-AVX512 $(ZNN_TUNE)                                 
KNC_FLAGS   =   -DZNN_KNC -mmic $(ZNN_TUNE)
CORES_FLAG  =   -DZNN_NUM_CORES=2

STANDARD_FLAG	=	-std=c++14
STD_FLAG	=

CXX             =   $(AT)icpc $(STANDARD_FLAG)                                           
OPTIMIZATION_FLAGS  =   -DNDEBUG -O3 -ipo -fp-model fast=2 -mkl=sequential -fno-rtti -fno-exceptions 
CS_FLAGS        =   -mkl=sequential                                                       
CS_LD_FLAGS     =   -mkl=sequential  -static-intel 

FPIC	=	-fPIC

CXXINCLUDES	=	-I$(BASE)/code/include -I$(BASE)/code/include/znn -I$(BASE)
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


ifneq ("$(wildcard $(O))","")
	LAYER_EXISTS=1
else
	LAYER_EXISTS=0
endif

FORCE:

./.tmp/%.o: ./.tmp/%.cpp FORCE
ifeq ($(LAYER_EXISTS),0)
	$(ECHO) "[CXX] compiling $<"
	$(ECHO) $(CXX) -c $(CXXFLAGS)  $(AVX2_FLAGS) $(PARAM_DEFS) -o $@ $<
	$(CXX)         -c $(CXXFLAGS)  $(AVX2_FLAGS) $(PARAM_DEFS) -o $@ $<
else
	$(ECHO)
endif

./.tmp/%.so: ./.tmp/%.o
ifeq ($(LAYER_EXISTS),0)
	$(ECHO) "[CXX] creating shared lib $@" 
	$(ECHO) $(CXX) $(DLFLAGS) $(CXXLDFLAGS) $(AVX2_FLAGS) -o $@ $< -ldl
	$(CXX)         $(DLFLAGS) $(CXXLDFLAGS) $(AVX2_FLAGS) -o $@ $< -ldl
	$(MV)   $@ $(O)
else 
	$(ECHO) 
endif

all: clean test_jit.bin 

test_jit.bin: FORCE
	$(ECHO) $(CXX) $(CXXFLAGS) $(AVX2_FLAGS) ./test/test_jit.cpp -o $@ 
	$(CXX) $(CXXFLAGS) $(AVX2_FLAGS) ./test/test_jit.cpp -o $@ 

clean:
	$(RM) -rf ./*.o ./*.so ./*.S *.bin 

remake: clean all


