#include "CublasManager.hpp"
#include <cstdio>

cublasHandle_t* cublasHandle = nullptr;

void cublasExitOnError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s",msg);
        fprintf(stderr, "Error code: %d \n", (int)status);
        exit(-1);
    }
}

void initCublas()
{
    if(cublasHandle != nullptr)
    {
        return;
    }

    cublasHandle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    cublasStatus_t status = cublasCreate(cublasHandle);
    cublasExitOnError(status, "Cublas initialization failed! \n");
    
#ifdef DEBUG
    status = cublasLoggerConfigure(true, 1,0,nullptr );
    cublasExitOnError(status, "Cublas could not start logging! \n");
#endif
}

void destroyCublas()
{
    if(cublasHandle != nullptr)
    {
        cublasStatus_t status = cublasDestroy(*cublasHandle);
        cublasExitOnError(status, "Cublas destruction failed! \n");
        delete cublasHandle;
        cublasHandle = nullptr;
    }
}
