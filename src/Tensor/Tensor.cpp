#include "Tensor.hpp"
#include <cuda_runtime_api.h>
#include "../Utils/error_logs.hpp"

unsigned char typeSizeTable[] = {2, 4, 8, 2, 4, 8, 2, 4, 8};


Tensor::Tensor(TensorShape dim, TensorType dtype)
:
tensorDeviceMemory(nullptr), dtype(dtype)
{
    this->shape = dim;
    rank = dim.size();
    if(rank != 0 )
    {
        unsigned int total_item_count = 1;
        for(auto& dimSize : dim)
        {
           logErrorAndExit(dimSize == 0, "Tensor cannot have dim of size 0! \n");
           total_item_count = dimSize * total_item_count;
        }
        cudaMalloc(&tensorDeviceMemory, total_item_count * typeSizeTable[(unsigned int)dtype]);
    }
    else
    {
        // rank 0 tensor ie scalar
         cudaMalloc(&tensorDeviceMemory, typeSizeTable[(unsigned int)dtype]);
    }
}

void Tensor::setTensor_HostToDevice(void* data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    logErrorAndExit(data == nullptr, "Copy source is unallocated tensor!\n");
    unsigned int tensor_byte_size = getNumberOfElements() * typeSizeTable[(unsigned int)dtype];
    cudaMemcpy(tensorDeviceMemory, data,  tensor_byte_size, cudaMemcpyHostToDevice);
}

void* Tensor::getTensorPointer()
{
    return tensorDeviceMemory;
}

unsigned int Tensor::getNumberOfElements()
{
    unsigned int total_size = 1;
    for(auto& dimSize : shape)
    {
        total_size *= dimSize;
    }

    return total_size;
}

void Tensor::setTensor_DeviceToDevice(void *data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    logErrorAndExit(data == nullptr, "Copy source is unallocated tensor!\n");
    cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyDeviceToDevice );
}

TensorShape Tensor::getShape()
{
    return shape;
}

char *Tensor::getTensorValues()
{
    return nullptr;
}

TensorType Tensor::getType()
{
    return dtype;
}
