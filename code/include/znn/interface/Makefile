HERE    	=       .
BINDIR		=	./bin

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
DATE    =   $(AT)date

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


CXX			=	$(AT)g++ -pthread $(STANDARD_FLAG)
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ffast-math -fno-omit-frame-pointer -fno-rtti -fno-exceptions
CS_FLAGS		=
CS_LD_FLAGS		=       -lpthread

FPIC	=	-fPIC
CXXINCLUDES	=	-I$(ZNNPHI_PATH)/include -I$(ZNNPHI_PATH)/include/znn -I$(ZNNPHI_PATH)/.. \
                -I/mnt/bucket/people/popovych/.julia/v0.5/CxxWrap/deps/src/cxx_wrap \
			    -I/mnt/bucket/labs/seung/seungmount/research/popovych/bins/julia/include/julia

CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas
CXXERR		=   -fmax-errors=5

HBW_LD_FLAG	=
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
DLFLAGS		= 	-fPIC -shared 

SO_NAME = libznnphiconv.so
%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(CXX) $(DEPFLAGS) $(AVX512_FLAGS) -MF $@ $<remake

%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(ECHO) $(CXX) -c $(CXXFLAGS) $(AVX512_FLAGS) -o $@ $<
	$(CXX) -c $(CXXFLAGS) $(AVX512_FLAGS) -o $@ $<

%.so: ./%.o
	$(ECHO) "[CXX] creasting shared lib $<"
	$(ECHO) $(CXX) $(DL_FLAGS)-o $(SO_NAME) $<
	$(CXX) $(DLFLAGS) -o $(SO_NAME) $<

#%.bin: ./%.o
#   $(ECHO) "[CXX] linking $^"
#	$(CXX) $^ $(CXXLDFLAGS) $(AVX512_FLAGS) -o $@

.PHONY: dl
dl: ZnnPhiConvLayer.so
	sudo cp $(SO_NAME) /usr/lib64/
all: clean dl
clean:
	$(RM) -rf ./*.o ./*.so ./*.S *.bin 

remake: clean all
clean:
	$(RM) *.o *.so *.S *.bin 

remake: clean all
