HERE    	=  .
OUTDIR 		= ./out

AT    		= @
DOLLAR   	= $$

CD	     	= $(AT)cd
CP			= $(AT)cp
ECHO		= @echo
CAT			= $(AT)cat
IF			= $(AT)if
LN			= $(AT)ln
MKDIR		= $(AT)mkdir
MV			= $(AT)mv
SED			= $(AT)sed
RM			= $(AT)rm -f
TOUCH		= $(AT)touch
CHMOD		= $(AT)chmod
DATE    	= $(AT)date
MKDIR_P  	= mkdir -p

# compilers
#ARCH 		= AVX2
ZNN_TUNE	=

SSE_FLAGS	=	-DZNN_SSE -msse $(ZNN_TUNE)
AVX_FLAGS	=	-DZNN_AVX -mavx $(ZNN_TUNE)
AVX2_FLAGS	=	-DZNN_AVX2 -march=core-avx2 $(ZNN_TUNE)
AVX512_FLAGS	= -DZNN_AVX512 -xCORE-AVX512 $(ZNN_TUNE)
KNC_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mfma $(ZNN_TUNE)

CORES_FLAG	=

STANDARD_FLAG	=	-std=c++14
STD_FLAG	=


CXX			= $(AT)icpc -pthread $(STANDARD_FLAG) -fopenmp
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -fno-omit-frame-pointer #-fno-rtti -fno-exceptions #-ffast-math 
CS_FLAGS	=
CS_LD_FLAGS	= -lpthread

FPIC		= -fPIC

MKL_INCLUDE_PATH =/opt/intel/mkl/include
CXXINCLUDES	=	-I./include -I./include/znn -I${MKL_INCLUDE_PATH}

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

LD_FLAGS    =  -lpthread -qopenmp  
#-L/opt/intel/mkl/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -Wl,--end-group -lmkl_avx2 -lznnphiconv 

PY_INTERFACE_FLAGS = `python3-config --cflags --ldflags` -Isrc/python/pybind11/include -fPIC 

ifeq ($(strip $(ARCH)),AVX512)
	ARCH_FLAGS=$(AVX512_FLAGS)
else ifeq ($(strip $(ARCH)),AVX2)
	ARCH_FLAGS=$(AVX2_FLAGS)
else
$(error Unsupported or unspecified architecture: ARCH==$(ARCH). Please set Makefile ARCH parameter to one of [AVX2, AVX512].) 
endif

.PHONY: bin, py, rerun, gen, FORCE

#NET_NAME = small_layer
#N = nets/${NET_NAME}.json
#W = nets/${NET_NAME}.h5

out/client.o: ./client.cpp 
	$(CXX) $(CXXFLAGS) -fPIC -c $^ -o $@

${OUTDIR}/generated/znet.cpp:  $(shell find ../python/pznet/generator -type f -name "*.py") $(N) $(W)
	rm -rf ${OUTDIR}
	${MKDIR_P} ${OUTDIR}
	${MKDIR_P} ${OUTDIR}/weights
	${MKDIR_P} ${OUTDIR}/generated
	python3 ../python/pznet/generator/main.py --net-path $(N) --weights-path $(W) \
		--out-path ${OUTDIR}/generated/znet.cpp \
		--architecture $(ARCH) --conv-cores $(CONV_CORES) --conv-ht $(CONV_HT) \
		--act-cores $(ACT_CORES) --act-ht $(ACT_HT) \
		--lin-cores $(LIN_CORES) --lin-ht $(LIN_HT) --cpu-offset $(CPU_OFFSET) \
		-O $(PZ_OPT) --ignore $(IGNORE) --time_each $(TIME_EACH)

${OUTDIR}/znet.o: ${OUTDIR}/generated/znet.cpp
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) -fPIC -c $^ -o $@

${OUTDIR}/znet_numpy_wrapper.o: main.cpp 
	$(CXX) $(CXXFLAGS) $(PY_INTERFACE_FLAGS) ${ARCH_FLAGS} -fPIC -c $^ -o $@

py: ${OUTDIR}/znet.o ${OUTDIR}/znet_numpy_wrapper.o 
	$(CXX) $(LD_FLAGS) $^ -shared -o ${OUTDIR}/znet.so 

ifneq ($(strip ${O}),)
	${MKDIR_P} ${O}
	${CP} ${OUTDIR}/znet.so $(O)/
	${CP} -r ${OUTDIR}/weights $(O)/
endif

bin: ${OUTDIR}/client.o ${OUTDIR}/znet.o 
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) $^ -o ${OUTDIR}/znet.bin
gen: ${OUTDIR}/generated/znet.cpp 
rerun: clean gen bin
	${OUTDIR}/znet.bin
rerun_py: clean gen py
	${OUTDIR}/client.py
all: znet 
clean:
	rm -rf ${OUTDIR}/*.so ${OUTDIR}/*.o ${OUTDIR}/*.bin
remake: clean all
