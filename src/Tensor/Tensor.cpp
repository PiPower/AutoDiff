#include "Tensor.hpp"
#include <cuda_runtime_api.h>
#include "../Utils/error_logs.hpp"

unsigned char typeSizeTable[] = {2, 4, 8, 2, 4, 8, 2, 4, 8};


Tensor::Tensor(TensorShape dim, TensorType dtype)
:
tensorDeviceMemory(nullptr), dtype(dtype)
{
    initCublas();
    initCudnn();

    logErrorAndExit(dtype != TensorType::float32, "currently usupported tensor type\n");
    this->shape = dim;
    rank = dim.size();
    cudaError err;
    if(rank != 0 )
    {
        unsigned int total_item_count = 1;
        for(auto& dimSize : dim)
        {
           logErrorAndExit(dimSize == 0, "Tensor cannot have dim of size 0! \n");
           total_item_count = dimSize * total_item_count;
        }

        err = cudaMalloc(&tensorDeviceMemory, total_item_count * typeSizeTable[(unsigned int)dtype]);
    }
    else
    {
        // rank 0 tensor ie scalar
        err =cudaMalloc(&tensorDeviceMemory, typeSizeTable[(unsigned int)dtype]);
    }
    logErrorAndExit(err != cudaSuccess, "Could not allocate memory for tensor on GPU");
    buildDescriptors();
}

void Tensor::setTensor_HostToDevice(void* data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    logErrorAndExit(data == nullptr, "Copy source is unallocated tensor!\n");
    cudaError err;
    err = cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyHostToDevice);
#ifdef DEBUG
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
#endif
}

void* Tensor::getTensorPointer()
{
    return tensorDeviceMemory;
}

DevicePointer *Tensor::getCudaDescriptorPointer()
{
    return cudaDescriptorDevice;
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
    cudaError err;
    err = cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyDeviceToDevice );
#ifdef DEBUG
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
#endif
}

TensorShape Tensor::getShape()
{
    return shape;
}

TensorType Tensor::getType()
{
    return dtype;
}

char *Tensor::getTensorValues()
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    char* data = new char[getNumberOfElements() * typeSizeTable[(unsigned int)dtype]];
    unsigned int tensor_byte_size = getNumberOfElements() * typeSizeTable[(unsigned int)dtype];
    cudaError err;
    err = cudaMemcpy(data, tensorDeviceMemory,  tensor_byte_size, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
#endif
    return data;
}

void Tensor::buildDescriptors()
{
    TensorDesc cudaDescriptor;
    cudaDescriptor.ndim = rank;
    unsigned int stride =1;
    for(int i = cudaDescriptor.ndim -1 ; i >=0; i--)
    {
        cudaDescriptor.dim[i] = shape[i];
        cudaDescriptor.dimStrides[i] = stride;
        stride *= cudaDescriptor.dim[i];
    }
    cudaError err;
    err = cudaMalloc((void**)&cudaDescriptorDevice, sizeof(TensorDesc));
    logErrorAndExit(err != cudaSuccess, "Could not allocate memory for tensor descriptor\n");
    err =cudaMemcpy(cudaDescriptorDevice, &cudaDescriptor, sizeof(TensorDesc), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "Could not set tensor descriptor on gpu side\n");

    tensorDescriptor = createTensorDescriptor(dtype, shape);
}

Tensor::~Tensor()
{
    cudaFree(tensorDeviceMemory);
}

void Tensor::addTensors(Tensor *dest, Tensor *left, Tensor *right)
{
    addTensorsOp((float*) dest->tensorDeviceMemory, (float*)left->tensorDeviceMemory, 
        (float*)right->tensorDeviceMemory, left->cudaDescriptorDevice, right->cudaDescriptorDevice);
}
