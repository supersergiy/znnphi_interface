#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>

void exec(const char* cmd) {
   FILE *fp;
   int status;
   const int PATH_MAX=10000;
   char path[PATH_MAX];

   fp = popen("ls *", "r");
   if (fp == NULL){
      printf ("Error opening file unexist.ent: %s\n",strerror(errno));
      return; 
   }

   while (fgets(path, PATH_MAX, fp) != NULL)
          printf("%s", path);

   //todo: check for failure
   status = pclose(fp);
   return; 
}
