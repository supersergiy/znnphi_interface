#pragma once
#include <sys/wait.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sstream>

#include <znn/layer/layer.hpp>
#define PYTHON_COMPILE_SCRIPT_REL_P "/code/include/znn/jit/jit.py"

#define DEBUG 1

namespace znn {
namespace phi {

void handleDLError(void)
{
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << dlsym_error << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void *jitGetHandle(std::string params)
{
   char dl_filename[1024];
   char *znnphi_path = getenv("ZNNPHI_PATH");

   std::stringstream compile_command;
   
   compile_command << znnphi_path << PYTHON_COMPILE_SCRIPT_REL_P << " " << params;
   
#ifdef DEBUG
   std::cout << compile_command.str() << std::endl;
#else
   compile_command << " 2> /dev/null"; 
#endif

   FILE* python_out = popen(compile_command.str().c_str(), "r");

   if (python_out == NULL) {
      std::cerr << "Calling jit.py fails\n";
      std::exit(EXIT_FAILURE);
   }

   fscanf(python_out, "%1024s", dl_filename);
#ifdef DEBUG
   std::cout << "dl_filename: " << dl_filename << std::endl; 
#endif

   void *handle = dlopen(dl_filename, RTLD_NOW);
   handleDLError();

   return handle;
}

Layer* jitMakeLayer(std::string layer_type, std::string layer_params)
{
   std::stringstream params;
   params << "LAYER=" << layer_type << " " << layer_params;

   void  *layer_handle = jitGetHandle(params.str()); 

   void* creator = dlsym(layer_handle, "createLayer");
   handleDLError();

   Layer *result = ((CreateLayer_fp)creator)(); 
   return result;
}

}
}
