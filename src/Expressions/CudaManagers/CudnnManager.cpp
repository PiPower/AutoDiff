#include "CudnnManager.hpp"
#include <cstdio>
#include <stdlib.h>

cudnnHandle_t* cudnnHandle = nullptr;

void cudnnExitOnError(cudnnStatus_t status, const char* msg)
{
    if(status != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s", msg);
        fprintf(stderr, "Error code: %d \n", (int)status);
        exit(-1);
    }
}

void initCudnn()
{
    if(cudnnHandle != nullptr)
    {
        return;
    }

    cudnnHandle = (cudnnHandle_t*)malloc(sizeof(cudnnHandle_t));
    cudnnStatus_t status = cudnnCreate(cudnnHandle);
    cudnnExitOnError(status, "Cudnn initialization failed! \n");

#ifdef DEBUG
    status = cudnnSetCallback(CUDNN_SEV_ERROR_EN, NULL, NULL );
    cudnnExitOnError(status, "Cudnn could not start logging! \n");
#endif
}

void destroyCudnn()
{
    if(cudnnHandle != nullptr)
    {
        cudnnStatus_t status = cudnnDestroy(*cudnnHandle);
        cudnnExitOnError(status, "Cudnn destruction failed! \n");
        delete cudnnHandle;
        cudnnHandle = nullptr;

    }
}

cudnnTensorDescriptor_t createTensorDescriptor(TensorType dtype, TensorShape shape)
{
    return cudnnTensorDescriptor_t();
}
