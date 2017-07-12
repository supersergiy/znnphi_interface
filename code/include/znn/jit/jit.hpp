#define _GLIBCXX_USE_CXX11_ABI 0
#include <sys/wait.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sstream>

#include <znn/layer/layer.hpp>
#define PYTHON_COMPILE_SCRIPT "python ./jit.py"

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
   std::stringstream compile_command;

   compile_command << PYTHON_COMPILE_SCRIPT << " " << params;
   compile_command << " 2> /dev/null"; // to silence the guy
   std::cout << compile_command.str() << std::endl;
   FILE* python_out = popen(compile_command.str().c_str(), "r");

   if (python_out == NULL) {
      std::cerr << "Calling jit.py fails\n";
      std::exit(EXIT_FAILURE);
   }

   fscanf(python_out, "%1024s", dl_filename);
   //std::cout << dl_filename << "!" << std::endl; 

   void *handle = dlopen(dl_filename, RTLD_NOW);
   handleDLError();

   return handle;
}

Layer* jitMakeLayer(std::string layer_type, std::string layer_params)
{
   std::stringstream params;
   params << "LAYER=" << layer_type << " " << layer_params;

   void  *layer_handle = jitGetHandle(params.str()); 

   Layer *result = ((CreateLayer_fp) dlsym(layer_handle, "createLayer"))();//TODO::what about layer deletion?
   std::cout << result << std::endl;
   handleDLError();
   return result;
}

}
}
