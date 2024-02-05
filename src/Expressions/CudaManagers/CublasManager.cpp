#include "CublasManager.hpp"
#include <cstdio>

cublasHandle_t* cublasHandle = nullptr;

void cudaExitOnError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, msg);
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
    cudaExitOnError(status, "Cuda initialization failed! \n");
}

void destroyCublas()
{
    if(cublasHandle != nullptr)
    {
        cublasStatus_t status = cublasDestroy(*cublasHandle);
        cudaExitOnError(status, "Cuda destruction failed! \n");
        delete cublasHandle;
    }
}
