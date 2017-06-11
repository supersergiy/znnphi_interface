#include<iostream>
#include<fstream>
#include<string>
using namespace std;

template<class T>
void readArrayFromFile(T *a, std::string path)
{
   ifstream InFile(path);
   
   if (!InFile) {
      std::cout << "Couldn't open " << path << std::endl;
      std::cout << "Make sure that proper weights folder is present at invocation location" << path << std::endl;
      exit(EXIT_FAILURE);
   }

   T e;
   size_t c = 0;
   while(InFile >> e) {
          a[c++] = e;
   }
}
