#include "error_logs.hpp"
#include <cstdio>
#include <stdlib.h>

void logErrorAndExit(bool condition, const char *message)
{
    if(condition)
    {
        fprintf(stderr,"%s \n" ,message);
        exit(-1); 
    }
}