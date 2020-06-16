#include<iostream>
#include<fstream>
#include<string>
using namespace std;

template<class T>
void readArrayFromFile(T *a, std::string path)
{
   // input file stream
   ifstream istrm(path);
   
   if (!istrm) {
      std::cout << "Couldn't open " << path << std::endl;
      std::cout << "Make sure that proper weights folder is present at invocation location" << std::endl;
      exit(EXIT_FAILURE);
   }

   T e;
   size_t pos = 0;
   while(istrm >> e) {
          a[pos++] = e;
   }
}
