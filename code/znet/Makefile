HERE    	=  .
BINDIR	=	./bin
NETDIR	=	./nets
CPPODIR	=  ./cpp_out

AT    	=	@
DOLLAR   = 	$$

JULIA    =  $(AT)julia
CD	     	=	$(AT)cd
CP			=	$(AT)cp
ECHO		=	@echo
CAT		=	$(AT)cat
IF			=	$(AT)if
LN			=	$(AT)ln
MKDIR		=	$(AT)mkdir
MV			=	$(AT)mv
SED		=	$(AT)sed
RM			=	$(AT)rm -f
TOUCH		=	$(AT)touch
CHMOD		=	$(AT)chmod
DATE    	=  $(AT)date


# compilers

ZNN_TUNE	=

SSE_FLAGS	=	-DZNN_SSE -msse $(ZNN_TUNE)
AVX_FLAGS	=	-DZNN_AVX -mavx $(ZNN_TUNE)
AVX2_FLAGS	=	-DZNN_AVX2 -mavx2 -mfma $(ZNN_TUNE)
AVX512_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mavx512cd -mfma $(ZNN_TUNE)
KNC_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mfma $(ZNN_TUNE)

CORES_FLAG	=

STANDARD_FLAG	=	-std=c++14
STD_FLAG	=


CXX			=	$(AT)g++ -pthread $(STANDARD_FLAG) -fopenmp
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ffast-math -fno-omit-frame-pointer #-fno-rtti -fno-exceptions
CS_FLAGS		=
CS_LD_FLAGS		=       -lpthread

FPIC	=	-fPIC
CXXINCLUDES	=	-I$(ZNNPHI_PATH)/code/include -I$(ZNNPHI_PATH)/code/include/znn -I$(ZNNPHI_PATH)/

CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas -Wno-unused-variable -Wno-return-type -Wno-unused-parameter
CXXERR		=   -fmax-errors=5

HBW_LD_FLAG	= #-lmemkind
HBW_FLAG	=	-DZNN_NO_HBW

DEPFLAGS	=	-MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)" $(CS_FLAGS) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG)
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG) $(STD_FLAG)

COMMON_FLAGS	=	-g $(INLINE_DEPFLAGS) $(CXXINCLUDES) $(CXXWARN) $(CXXERR) \
					$(HBW_FLAG) $(CORES_FLAG)

DBG_FLAGS		=	$(COMMON_FLAGS) $(CS_FLAGS)
LD_OPTIMIZATION_FLAGS	=	-DNDEBUG -O3

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)

COMMON_LDFLAGS		=	$(HBW_LD_FLAG)

DBG_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) -g
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) $(LD_OPTIMIZATION_FLAGS)

#CXXFLAGS	=	$(DBG_FLAGS)
CXXFLAGS	=	$(OPT_FLAGS)
CXXLDFLAGS	=	$(OPT_LDFLAGS)

LD_FLAGS    =   -L$(ZNNPHI_PATH)/code/include/znn/interface \
				-lznnphiconv


$(CPPODIR)/%_b${B}.cpp: run.jl $(NETDIR)/%.json $(NETDIR)/%.h5
	$(ECHO) "[Generating $@]..."
	$(JULIA) $^ ${B} ${N} > $@ 

$(BINDIR)/%_b${B}.bin: $(CPPODIR)/%_b${B}.cpp
	$(ECHO) "[Generating $@]..."
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) $(LD_FLAGS) $< -o $@


/usr/lib/znnphiconvlib.so:
	#Nothin

.PRECIOUS: %.cpp 
.PHONY: unet, all, layers, znet, client
test: tmp.cpp
	$(ECHO) $(CXX) -c $(CXXFLAGS) $(AVX_FLAGS) tmp.cpp
	$(CXX)  $(CXXFLAGS) $(AVX_FLAGS) $(LD_FLAGS) tmp.cpp 

client:
znet: cpp_out/znet.cpp cpp_out/client.cpp 
	$(ECHO) $(CXX) -c $(CXXFLAGS) $(AVX_FLAGS) $^ 
	$(CXX)  $(CXXFLAGS) $(AVX_FLAGS) $(LD_FLAGS) $^ 

layers: 
	julia run_separate.jl ./nets/unet.json ./nets/unet.h5 ${B} 1 
	make layer_{1..49} B=${B} #-j64
layer_%: $(BINDIR)/layer_%_b${B}.bin
	$^
unet: $(BINDIR)/unet_b${B}.bin
	numactl --membind=1 $^
	tail bench/timings${B}.txt
small_layer: $(BINDIR)/small_layer_b${B}.bin
	numactl --membind=1 $^
big_layer: $(BINDIR)/big_layer_b${B}.bin
	numactl --membind=1 $^
all:  unet
clean:
	rm -rf ./cpp_out/ ./bin/

remake: clean all
