#include<iostream>
#include<fstream>
#include<string>
using namespace std;

template<class T>
void readArrayFromFile(T *a, const char *path)
{
   ifstream InFile(path);
   
   T e;
   size_t c = 0;
   while(InFile >> e) {
          a[c++] = e;
   }
}
