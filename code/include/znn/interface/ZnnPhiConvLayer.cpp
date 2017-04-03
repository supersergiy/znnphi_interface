#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
#include <sstream>
#include <iterator>
#include <string.h>

#include "ZnnPhiConvLayer.hpp"

#define MAX_STRING 512
#define COMPILE_CONV_WRAPPER_SCRIPT "include/znn/interface/generate_dl.sh"
#define BASH "/bin/bash"

namespace znn
{
namespace phi
{

std::string generateParamString(
                         int bn, int ifm, int ofm, int id,
                         int ihw, int kd, int khw,
                         int padd, int padhw,
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
    params.push_back(std::to_string(cores));
    params.push_back(std::to_string(ht));
    
    std::copy(params.begin(), params.end(), 
              std::ostream_iterator<std::string>(param_string, delim)); 
    return param_string.str();
}

std::string generateWrapperDLName(
                        int bn, int ifm, int ofm, int id,
                        int ihw, int kd, int khw,
                        int padd, int padhw,
                        int cores, int ht)
{
    const char delim[] = "_";
    std::ostringstream generated_name;
    std::string param_string;
    
    const char *znnphi_path = std::getenv("ZNNPHI_PATH");

    param_string = generateParamString(bn, ifm, ofm, id, ihw, kd, khw, 
                                       padd, padhw, cores, ht, delim);

    generated_name << znnphi_path << "/include/znn/interface/dl_files/";
    generated_name << "conv_wrapper_" << param_string << ".so";

    return generated_name.str();
}

std::string generateCompileDLCommand(std::string& dl_filename,
                        int bn, int ifm, int ofm, int id,
                        int ihw, int kd, int khw,
                        int padd, int padhw,
                        int cores, int ht)
{
    const char delim[] = " ";
    std::ostringstream command_string;
    std::string param_string;

    param_string = generateParamString(bn, ifm, ofm, id, ihw, kd, khw, 
                                       padd, padhw, cores, ht, delim);
    
    const char *znnphi_path = std::getenv("ZNNPHI_PATH");

    command_string << BASH << " " ;
    command_string << znnphi_path << "/" << COMPILE_CONV_WRAPPER_SCRIPT << " ";
    command_string << dl_filename << " " << param_string;
     
    return command_string.str(); 
}

void handleDLError(void)
{
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << dlsym_error << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void *loadConvWrapperDL(int bn, int ifm, int ofm, int id,
                        int ihw, int kd, int khw,
                        int padd, int padhw,
                        int cores, int ht)
{
    std::string dl_filename;
    std::string compile_command;
    
    dl_filename = generateWrapperDLName(bn, ifm, ofm, id, ihw, kd, khw, padd, 
                                        padhw, cores, ht);
    compile_command = generateCompileDLCommand(dl_filename,
                          bn, ifm, ofm, id, ihw, kd, khw, padd, padhw, cores, ht);

    std::system(compile_command.c_str());
    
    void *handle = dlopen(dl_filename.c_str(), RTLD_NOW);
    handleDLError();
    return handle;
}


ZnnPhiConvLayer::ZnnPhiConvLayer(int bn, int ifm, int ofm, int id,
                                 int ihw, int kd, int khw,
                                 int padd, int padhw,
                                 int cores, int ht)
{

    void *wrapper_handle = loadConvWrapperDL(bn, ifm, ofm, id, ihw, kd, khw, 
                                             padd, padhw, cores, ht);
    createConvWrapper = (CreateConvWrapper_fp) dlsym(wrapper_handle, "createConvWrapper");
    handleDLError();
    destroyConvWrapper = (DestroyConvWrapper_fp) dlsym(wrapper_handle, "destroyConvWrapper");
    handleDLError();
    conv_wrapper = createConvWrapper();
}

ZnnPhiConvLayer::~ZnnPhiConvLayer()
{
    destroyConvWrapper(conv_wrapper);
}

void ZnnPhiConvLayer::forward(float *in, float *out, float *ker, float *bi)
{
    conv_wrapper->compute(in, out, ker, bi);
}

}
}
