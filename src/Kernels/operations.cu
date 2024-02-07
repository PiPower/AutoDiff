#include "kernel_api.h"

/*
all tensors must be Fully-Packed Tensors (https://docs.nvidia.com/deeplearning/cudnn/developer/core-concepts.html)
tensor dest MUST have the same dimensions as left.
each dimension x of left must satisfy x = c_i*y where y is 
corresponding dimension of right tensor and c_i > 0 is constant whole number for ith dimension
If ubove conditions are not met, function has undefined behaviour
*/

__device__ void _resolveOffset(unsigned int* memoryLocation, TensorDesc* leftDesc, TensorDesc* rightDesc, unsigned int* resultOffset)
{
    unsigned int currentMemLoc = *memoryLocation;
    *resultOffset = 0;
    for(int i = 0; i < leftDesc->ndim ; i++)
    {   
        //find i-th dim in left tensor
        unsigned int leftDim = currentMemLoc/leftDesc->dimStrides[i];
        //change memory location in left(larger) tensor
        memoryLocation = memoryLocation - leftDim * leftDesc->dimStrides[i]; 
        //find aligned offset in right(smaller) tensor
        *resultOffset += (leftDim%rightDesc->dim[i] ) * rightDesc->dimStrides[i];
    }
}


__global__ void _kernelAddTensors(float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        unsigned int rightOffset = 0;
        _resolveOffset(&threadIndex, leftDesc, rightDesc, &rightOffset);
        dest[threadIndex] = left[threadIndex] + right[rightOffset]; 

        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}





extern "C" void addTensors( float* dest, float* left, float* right, 
        TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    _kernelAddTensors<<<16,16>>>(dest, left, right, leftDesc, rightDesc);
    cudaDeviceSynchronize();
}


