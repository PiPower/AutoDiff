#include "Tensor.hpp"
#include <cuda_runtime_api.h>
#include "../Utils/error_logs.hpp"

unsigned char typeSizeTable[] = {16, 32, 64, 16, 32, 64, 16, 32, 64};


Tensor::Tensor(std::vector<unsigned int> dim, TensorType dtype)
:
tensorDeviceMemory(nullptr), dtype(dtype)
{
    this->shape = dim;
    if(dim.size() != 0 )
    {
        unsigned int total_size = 0;
        for(auto& dimSize : dim)
        {
           logErrorAndExit(dimSize == 0, "Tensor cannot have dim of size 0! \n");
           total_size += dimSize;
        }
        cudaMalloc(&tensorDeviceMemory, total_size * typeSizeTable[(unsigned int)dtype]);
    }
}

void Tensor::setTensor_HostToDevice(void* data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy source is unexisting host memory!\n");
    cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyHostToDevice );
}

void* Tensor::getTensorPointer()
{
    return tensorDeviceMemory;
}

unsigned int Tensor::getNumberOfElements()
{
    unsigned int total_size = 0;
    for(auto& dimSize : shape)
    {
        total_size += dimSize;
    }

    return total_size;
}

void Tensor::setTensor_DeviceToDevice(void *data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy source is unexisting device memory!\n");
    cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyDeviceToDevice );
}

TensorShape Tensor::getShape()
{
    return shape;
}

TensorType Tensor::getType()
{
    return dtype;
}
