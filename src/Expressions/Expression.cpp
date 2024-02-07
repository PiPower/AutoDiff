#include "Expression.hpp"

void Expression::buildResultCudaDesc()
{
    TensorDesc cudaDescriptor;
    cudaDescriptor.ndim = result->getShape().size();
    unsigned int stride =1;
    for(int i = cudaDescriptor.ndim -1 ; i >=0; i--)
    {
        cudaDescriptor.dim[i] = result->getShape()[i];
        cudaDescriptor.dimStrides[i] = stride;
        stride *= cudaDescriptor.dim[i];
    }
    cudaError_t err;
    err = cudaMalloc(&cudaDescriptorDevice, sizeof(TensorDesc));
    logErrorAndExit(err != cudaSuccess, "Could not allocate memory for tensor descriptor\n");
    err =cudaMemcpy(cudaDescriptorDevice, &cudaDescriptor, sizeof(TensorDesc), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "Could not set tensor descriptor on gpu side\n");
}

Expression::Expression()
    : visited(false), result(nullptr), addedToExecutionList(false),
      tensorDescriptor(nullptr), cudaDescriptorDevice(nullptr)
{
    initCublas();
    initCudnn();
}

Expression::~Expression()
{
}