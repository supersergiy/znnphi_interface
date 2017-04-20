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

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib64

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
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ffast-math -fno-omit-frame-pointer #-fno-rtti -fno-exceptions
CS_FLAGS		=
CS_LD_FLAGS		=       -lpthread

FPIC	=	-fPIC
CXXINCLUDES	= -I$(ZNNPHI_PATH)/include -I$(ZNNPHI_PATH)/include/znn -I$(ZNNPHI_PATH)/.. \
				  -I$(PYTHON_INCLUDE) \
				  -I$(BOOST_INC)
				  
LD_FLAGS	= 	-L/usr/local/lib64 -lstdc++ \
				-L$(BOOST_LIB) -lboost_python \
				-L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION)
CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-parameter -Wno-return-type
CXXERR		=   -fmax-errors=5

HBW_LD_FLAG	=
HBW_FLAG	=	-DZNN_NO_HBW

DEPFLAGS	=	-MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)" $(CS_FLAGS) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG)
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG) $(STD_FLAG)

COMMON_FLAGS	=	-g $(INLINE_DEPFLAGS) $(CXXINCLUDES) $(LD_FLAGS) $(CXXWARN) $(CXXERR) \
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
		-c -ldl $(CXXFLAGS) $(AVX_FLAGS) -o ZnnPhiConvLayer.o
layer_dl:
	$(CXX) ZnnPhiConvLayer.cpp -c -fPIC $(CXXFLAGS) $(AVX_FLAGS) -o ZnnPhiConvLayer.o
	$(ECHO) -c -fPIC $(CXXFLAGS) $(AVX_FLAGS) -o ZnnPhiConvLayer.o
	$(CXX) -shared -Wl,--export-dynamic -L/usr/local/lib64 -lstdc++ $(LD_FLAGS) -o libznnphiconv.so  ZnnPhiConvLayer.o
clean:
	$(RM) *.o *.so *.S *.bin 

remake: clean all
