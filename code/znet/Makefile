HERE    	=  .
BINDIR	=	./bin
NETDIR	=	./nets

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
MKDIR_P  = mkdir -p

# compilers

ZNN_TUNE	=

SSE_FLAGS	=	-DZNN_SSE -msse $(ZNN_TUNE)
AVX_FLAGS	=	-DZNN_AVX -mavx $(ZNN_TUNE)
AVX2_FLAGS	=	-DZNN_AVX2 -march=core-avx2 $(ZNN_TUNE)
AVX512_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mavx512cd -mfma $(ZNN_TUNE)
KNC_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mfma $(ZNN_TUNE)

CORES_FLAG	=

STANDARD_FLAG	=	-std=c++14
STD_FLAG	=


CXX			=	$(AT)icpc -pthread $(STANDARD_FLAG) -fopenmp
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -fno-omit-frame-pointer #-fno-rtti -fno-exceptions #-ffast-math 
CS_FLAGS		=
CS_LD_FLAGS		=       -lpthread

FPIC	=	-fPIC
CXXINCLUDES	=	-I$(ZNNPHI_PATH)/code/include -I$(ZNNPHI_PATH)/code/include/znn -I$(ZNNPHI_PATH)/ -I./src/

#CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas -Wno-unused-variable -Wno-return-type -Wno-unused-parameter
CXXERR		=   -fmax-errors=5

HBW_LD_FLAG	= #-lmemkind
HBW_FLAG	=	-DZNN_NO_HBW

#DEPFLAGS	=	-MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)" $(CS_FLAGS) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG)
#INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T) $(HBW_FLAG) $(CORES_FLAG) $(STD_FLAG)

COMMON_FLAGS	=	$(INLINE_DEPFLAGS) $(CXXINCLUDES) $(CXXWARN) $(CXXERR) \
					$(HBW_FLAG) $(CORES_FLAG) -DZNN_NO_CUDA 

DBG_FLAGS		=	$(COMMON_FLAGS) $(CS_FLAGS)
LD_OPTIMIZATION_FLAGS	=	-DNDEBUG -O3

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)

COMMON_LDFLAGS		=	$(HBW_LD_FLAG)

DBG_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) -g
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) $(LD_OPTIMIZATION_FLAGS)

#CXXFLAGS	=	$(DBG_FLAGS)
CXXFLAGS	=	$(OPT_FLAGS)
CXXLDFLAGS	=	$(OPT_LDFLAGS)

LD_FLAGS    =   -lznnphiconv

PY_INTERFACE_FLAGS = `python-config --cflags --ldflags` -Isrc/python/pybind11/include -fPIC 

.PHONY: bin, py, rerun, gen, FORCE

NET_NAME = unet
N = nets/${NET_NAME}.json
W = nets/${NET_NAME}.h5

src/generated/znet.cpp: 
	rm -rf out
	${MKDIR_P} out
	${MKDIR_P} out/weights
	python src/znet_generator/main.py $(N) $(W) src/generated/znet.cpp
out/client.o: src/client.cpp 
	$(CXX) $(CXXFLAGS) -fPIC -c $^ -o $@
out/znet.o: src/generated/znet.cpp 
	$(CXX) $(CXXFLAGS) -fPIC -c $^ -o $@
out/znet_numpy_wrapper.o: src/python/znet_numpy_wrapper.cpp 
	$(CXX) $(CXXFLAGS) $(PY_INTERFACE_FLAGS) -fPIC -c $^ -o $@
py: out/znet.o out/znet_numpy_wrapper.o 
	$(CXX) $^ -lznnphiconv -shared -o out/znet.so 
bin: out/client.o out/znet.o 
	$(CXX) $(CXXFLAGS) $(AVX2_FLAGS) $(LD_FLAGS) $^ -o ./out/znet.bin
gen: src/generated/znet.cpp FORCE
rerun: clean gen bin
	./out/znet.bin
all: znet 
clean:
	rm -rf out/*.so out/*.o out/*.bin
remake: clean all
