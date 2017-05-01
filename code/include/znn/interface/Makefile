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

$(BINDIR)/avx/obj/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) $(AVX_FLAGS) -o $@ $<
	#$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

$(BINDIR)/avx/obj/%.S: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -S $(CXXFLAGS) $(AVX_FLAGS) -o $@ $<

$(BINDIR)/avx/%.bin: $(BINDIR)/avx/obj/%.o
	$(ECHO) "[CXX] linking $^"
	$(ECHO) $(CXX) $^ $(CXXLDFLAGS) $(AVX_FLAGS) -o $@
	$(CXX) $^ $(CXXLDFLAGS) $(AVX_FLAGS) -o $@


.PHONY: layer test layer_dl 
test:
	$(CXX) ZnnPhiConvWrapper.cpp \
		$(DLFLAGS) $(CXXFLAGS) $(AVX_FLAGS) -o $(DL_NAME) 
layer:
	$(CXX) ZnnPhiConvLayer.cpp \
		-c -ldl $(CXXFLAGS) $(AVX512_FLAGS) -o ZnnPhiConvLayer.o
layer_dl:
	$(CXX) ZnnPhiConvLayer.cpp \
		-c -ldl -fPIC $(CXXFLAGS) $(AVX512_FLAGS) -o ZnnPhiConvLayer.o

	$(CXX) -shared -ldl -o libznnphiconv.so  ZnnPhiConvLayer.o
	sudo cp libznnphiconv.so /usr/lib64/
clean:
	$(RM) *.o *.so *.S *.bin 

remake: clean all
