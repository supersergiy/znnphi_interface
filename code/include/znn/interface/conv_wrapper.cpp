
#define _GLIBCXX_USE_CXX11_ABI 0
#include <iostream>
#include <stdlib.h>
#include <dlfcn.h>
#include <vector>
#include <sstream>
#include <iterator>
#include <string.h>
#include <chrono>

#include "conv_wrapper.hpp"

#define MAX_STRING 1024 
#define BASH "/bin/bash"
#define RELATIVE_MAKEFILE_PATH "code/include/znn/interface"

namespace znn
{
namespace phi
{

std::string generateParamString(
                         int bn, int ifm, int ofm, int id,
                         int ihw, int kd, int khw,
                         int padd, int padhw, bool Activation,
                         int cores, int ht,
                         const char* delim)
{
    std::ostringstream param_string;
    std::vector<std::string> params;
    
    params.push_back(std::to_string(bn));
    params.push_back(std::to_string(ifm));
    params.push_back(std::to_string(ofm));
    params.push_back(std::to_string(id));
    params.push_back(std::to_string(ihw));
    params.push_back(std::to_string(kd));
    params.push_back(std::to_string(khw));
    params.push_back(std::to_string(padd));
    params.push_back(std::to_string(padhw));
    params.push_back(std::to_string(Activation));
    params.push_back(std::to_string(cores));
    params.push_back(std::to_string(ht));
    
    std::copy(params.begin(), params.end(), 
              std::ostream_iterator<std::string>(param_string, delim)); 
    return param_string.str();
}

std::string generateLayerSOName(
                        int bn, int ifm, int ofm, int id,
                        int ihw, int kd, int khw,
                        int padd, int padhw, bool Activation,
                        int cores, int ht)
{
    const char delim[] = "_";
    std::ostringstream generated_name;
    std::string param_string;
    
    const char *znnphi_path = std::getenv("ZNNPHI_PATH");

    param_string = generateParamString(bn, ifm, ofm, id, ihw, kd, khw, 
                                       padd, padhw, Activation, cores, ht, delim);

    generated_name << znnphi_path << "/lib/";
    generated_name << "conv_layer_" << param_string << ".so";

    return generated_name.str();
}

std::string generateCompileSOCommand(std::string& dl_filename,
                        int bn, int ifm, int ofm, int id,
                        int ihw, int kd, int khw,
                        int padd, int padhw, bool Activation,
                        int cores, int ht)
{
    std::ostringstream command_string;
    const char *znnphi_path = std::getenv("ZNNPHI_PATH");

    //invoke makefile in the interface folder
    command_string << "make -s -C " << znnphi_path << "/" << RELATIVE_MAKEFILE_PATH;
   
    //specify the target: layer
    command_string << " layer";

    //specify layer params 
    command_string << " BN=" << bn << " IFM=" << ifm << " OFM=" << ofm;
    command_string << " ID=" << id << " IHW=" << ihw; 
    command_string << " KD=" << kd << " KHW=" << khw;
    command_string << " PADD=" << padd << " PADHW=" << padhw; 
    command_string << " CORES=" << cores << " HT=" << ht; 
    command_string << " ACTIVATION=" << Activation;
    return command_string.str(); 
}

void handleSOError(void)
{
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << dlsym_error << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void *loadConvLayerSO(int bn, int ifm, int ofm, int id,
                        int ihw, int kd, int khw,
                        int padd, int padhw, bool Activation,
                        int cores, int ht)
{
    std::string dl_filename;
    std::string compile_command;
    
    dl_filename = generateLayerSOName(bn, ifm, ofm, id, ihw, kd, khw, padd, 
                                      padhw, Activation, cores, ht);

    compile_command = generateCompileSOCommand(dl_filename,
                          bn, ifm, ofm, id, ihw, kd, khw, padd, padhw, Activation, cores, ht);
    //compile_command += " &>/dev/null; "; 

    //std::cout << (compile_command.c_str());

    auto begin = std::chrono::high_resolution_clock::now();  
    std::system(compile_command.c_str());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                  (end - begin).count();
    double secs = static_cast<double>(duration) / 1000000;

    std::string param = generateParamString(bn, ifm, ofm, id, ihw, kd, khw,
                                       padd, padhw, Activation, cores, ht, "_");
    void *handle = dlopen(dl_filename.c_str(), RTLD_NOW);
    handleSOError();

    return handle;
}

 
//TODO: add checkf for null pointers with uninitialized layers
//void ConvWrapper::init(int bn, int ifm, int ofm, int id,
ConvWrapper::ConvWrapper(int bn, int ifm, int ofm, int id,
                         int ihw, int kd, int khw,
                         int padd, int padhw, bool Activation,
                         int cores, int ht)
{
    void *layer_handle = loadConvLayerSO(bn, ifm, ofm, id, ihw, kd, khw, 
                                             padd, padhw, Activation, cores, ht);
    createConvLayer = (CreateConvLayer_fp) dlsym(layer_handle, "createConvLayer");
    handleSOError();
    destroyConvLayer = (DestroyConvLayer_fp) dlsym(layer_handle, "destroyConvLayer");
    handleSOError();
    conv_layer = createConvLayer();
}

ConvWrapper::~ConvWrapper()
{
    destroyConvLayer(conv_layer);
}

void ConvWrapper::forward(float const* __restrict in, float *out, 
                          float const* __restrict ker, 
                          float const* __restrict bi, 
                          float const* __restrict scale)
{
    conv_layer->compute(in, out, ker, bi, scale);
}

}
}
