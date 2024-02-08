#include "CudnnManager.hpp"
#include <cstdio>
#include <stdlib.h>
#include "../../Utils/error_logs.hpp"
#include <iostream>
cudnnHandle_t* cudnnHandle = nullptr;
DevicePointer* workSpaceDvcPointer = nullptr;
unsigned int workSpaceSize = 100000;

cudnnDataType_t getDataType(TensorType dtype)
{
    switch (dtype)
    {
        case TensorType::float16:
            return CUDNN_DATA_BFLOAT16;
        case TensorType::float32:
            return CUDNN_DATA_FLOAT;
        case TensorType::float64:
            return CUDNN_DATA_DOUBLE;
        case TensorType::int32:
            return CUDNN_DATA_INT32;
        case TensorType::int64:
            return CUDNN_DATA_INT64;
        default:
            fprintf(stderr, "Unsported dtype");
            exit(-1);
    }
}

int getDataTypeStride(TensorType dtype)
{
    switch (dtype)
    {
        case TensorType::float16:
            return 2;
        case TensorType::float32:
            return 4;
        case TensorType::float64:
            return 8;
        case TensorType::int32:
            return 4;
        case TensorType::int64:
            return 8;
        default:
            fprintf(stderr, "Unsported dtype");
            exit(-1);
    }
}


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
//for logging to work CUDNN_LOGDEST_DBG MUST be set to desired output: stdout or stderr or file
    status = cudnnSetCallback(0x0F, NULL, NULL );
    cudnnExitOnError(status, "Cudnn could not start logging! \n");
#endif
    cudaError err;
    err = cudaMalloc(&workSpaceDvcPointer, workSpaceSize);
    logErrorAndExit(err != cudaSuccess, "could not allocate memory for cudnn workspace");
}

void destroyCudnn()
{
    if(cudnnHandle != nullptr)
    {
        cudnnStatus_t status = cudnnDestroy(*cudnnHandle);
        cudnnExitOnError(status, "Cudnn destruction failed! \n");
        delete cudnnHandle;
        cudnnHandle = nullptr;
        cudaFree(workSpaceDvcPointer);
    }
}

void* createCudnnDescriptor(TensorType dtype, TensorShape shape)
{
    cudnnTensorDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&desc);
    cudnnExitOnError(status, "Cudnn tensor descriptor failed! \n");

    int dimCount = shape.size() > 3 ? shape.size() : 4;
    int* dim = new int[dimCount];
    int* dimStride = new int[dimCount];

    int stride = 1;
    int z = shape.size();
    for(int i =  dimCount - 1, j = shape.size()-1; i >= 0; i--, j--)
    {
        dim[i] =  j >= 0? shape[j] : 1;
        dimStride[i] = stride;
        stride = stride * dim[i];
    } 

    status = cudnnSetTensorNdDescriptor(desc, getDataType(dtype), dimCount, dim, dimStride);
    cudnnExitOnError(status, "Cudnn tensor descriptor set failed! \n");
    delete[] dim;
    delete[] dimStride;

    return desc;
}

void destroyCudnnDescriptor(void *descriptor)
{
    cudnnStatus_t status;
    status = cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t )descriptor);
#ifdef DEBUG
    cudnnExitOnError(status, "Cudnn could not destroy descriptor \n");
#endif
}

void addTensors(const void *alpha,
                const void* OperandDesc,  DevicePointer* OperandDevice,
                const void *beta,const void* DestinationDesc, DevicePointer *DestinationDevice)
{
    cudnnStatus_t status;
    status = cudnnAddTensor(*cudnnHandle, alpha, (cudnnTensorDescriptor_t)OperandDesc,
    OperandDevice, beta, (cudnnTensorDescriptor_t)DestinationDesc, DestinationDevice);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudnnExitOnError(status, "Cudnn could not start logging! \n");
#endif
}

cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(cudnnReduceTensorOp_t reduce_op)
{
    cudnnReduceTensorDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateReduceTensorDescriptor(&desc); 
    cudnnExitOnError(status, "Cudnn tensor reduce descriptor failed! \n");
    status = cudnnSetReduceTensorDescriptor(desc, reduce_op, 
    CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES );

    return desc;
}

void reduceTensors(const cudnnReduceTensorDescriptor_t reduceTensorDesc,  
                    const void *alpha, DevicePointer *Operand, const void* OperandDesc,
                    const void *beta, const void* DestinationDesc, DevicePointer *Destination)
{
    cudnnReduceTensor(*cudnnHandle, reduceTensorDesc,nullptr,0,workSpaceDvcPointer,
    workSpaceSize,alpha,(cudnnTensorDescriptor_t)OperandDesc, Operand,
    beta, (cudnnTensorDescriptor_t)DestinationDesc, Destination);

    cudaDeviceSynchronize();
}