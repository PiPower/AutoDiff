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

__global__ void _kernelSubtractTensors(float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        unsigned int rightOffset = 0;
        _resolveOffset(&threadIndex, leftDesc, rightDesc, &rightOffset);
        dest[threadIndex] = left[threadIndex] - right[rightOffset]; 

        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelMulTensors(float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        unsigned int rightOffset = 0;
        _resolveOffset(&threadIndex, leftDesc, rightDesc, &rightOffset);
        dest[threadIndex] = left[threadIndex] * right[rightOffset]; 

        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelDivideTensors(float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        unsigned int rightOffset = 0;
        _resolveOffset(&threadIndex, leftDesc, rightDesc, &rightOffset);
        dest[threadIndex] = left[threadIndex] / right[rightOffset]; 

        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

/*
lets assume we have tensor x{i_1, i_2, ... , i_k} and y{a_1 * i_1, a_2* i_2, ... , a_k* i_k} 
where  a_1, a_2, ... , a_k > 0 are ints and k <=5
op below is used to sum over aligned axes
simple example
x{5} and y{20} 
in order to accumulate y into x we have 
x[1] = y[1] + y[6]+ y[11]+ y[16]
.
.
.
x[5] = y[5] + y[10]+ y[15]+ y[20]
Disclaimer:
current implementation is quire slow due to atomicAdd
if possible should be avoided
*/
__global__ void _kernelAxisAlignedAccumulation(float* dest, float* src, TensorDesc* destDesc, TensorDesc* srcDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = srcDesc->dim[0] * srcDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        unsigned int rightOffset = 0;
        _resolveOffset(&threadIndex, srcDesc, destDesc, &rightOffset);
        atomicAdd( dest + rightOffset, src[threadIndex] );
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

// dim_dest == dim_operand
__global__ void _kernelScaleByConstant(float* dest, float* operand, float* scalar, TensorDesc* leftDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = (*scalar) * (operand[threadIndex]); 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelAddConstant(float* dest, float* operand, float* scalar, TensorDesc* leftDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = (*scalar) + (operand[threadIndex]); 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelDivideByConstant(float* dest, float* operand, float* scalar, TensorDesc* leftDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = (operand[threadIndex]) / *scalar; 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelSqrt(float* dest, float* operand, TensorDesc* leftDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = sqrt(operand[threadIndex]); 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelExp(float* dest, float* operand, TensorDesc* leftDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = exp(operand[threadIndex]); 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelLog(float* dest, float* operand, TensorDesc* leftDesc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = leftDesc->dim[0] * leftDesc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = log(operand[threadIndex]); 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

// y*log(p) op fusion for crossentropy, all tensors must have the same dimensions
__global__ void _kernelCCfusionForward(float* dest, float* predictions, float* labels, TensorDesc* desc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = desc->dim[0] * desc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = -1.0f * labels[threadIndex] * log(predictions[threadIndex]); 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

__global__ void _kernelCCfusionBackward(float* dest, float* predictions, float* labels, TensorDesc* desc)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int upper_memory_bound = desc->dim[0] * desc->dimStrides[0];
    while (threadIndex < upper_memory_bound)
    {
        dest[threadIndex] = (-1.0f / predictions[threadIndex]) * labels[threadIndex] ; 
        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }
}

extern "C" void addTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    _kernelAddTensors<<<16,16>>>(dest, left, right, leftDesc, rightDesc);
    cudaDeviceSynchronize();
}

extern "C" void subtractTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    _kernelSubtractTensors<<<16,16>>>(dest, left, right, leftDesc, rightDesc);
    cudaDeviceSynchronize();
}
extern "C" void mulTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    _kernelMulTensors<<<16,16>>>(dest, left, right, leftDesc, rightDesc);
    cudaDeviceSynchronize();
}

extern "C" void divideTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc)
{
    _kernelDivideTensors<<<16,16>>>(dest, left, right, leftDesc, rightDesc);
    cudaDeviceSynchronize();
}

extern "C" void axisAlignedAccumulationOp( float* dest, float* src, TensorDesc* destDesc, TensorDesc* srcDesc)
{
    _kernelAxisAlignedAccumulation<<<16,16>>>(dest, src, destDesc, srcDesc);
    cudaDeviceSynchronize();
}

extern "C" void addConstantOp(float* dest, float* operand, float* scalar, TensorDesc* leftDesc)
{
    _kernelAddConstant<<<16,16>>>(dest, operand, scalar, leftDesc);
    cudaDeviceSynchronize();
}

extern "C" void divideByConstantOp(float* dest, float* operand, float* scalar, TensorDesc* leftDesc)
{
    _kernelDivideByConstant<<<16,16>>>(dest, operand, scalar, leftDesc);
    cudaDeviceSynchronize();
}

extern "C" void sqrtOp(float* dest, float* operand, TensorDesc* leftDesc)
{
    _kernelSqrt<<<16,16>>>(dest, operand, leftDesc);
    cudaDeviceSynchronize();
}

extern "C" void scaleByConstantOp(float* dest, float* operand, float* scalar, TensorDesc* leftDesc)
{
    _kernelScaleByConstant<<<16,16>>>(dest, operand, scalar, leftDesc);
    cudaDeviceSynchronize();
}

extern "C" void expOp(float* dest, float* operand, TensorDesc* leftDesc)
{
    _kernelExp<<<16,16>>>(dest, operand, leftDesc);
    cudaDeviceSynchronize();
}

extern "C" void logOp(float* dest, float* operand, TensorDesc* leftDesc)
{
    _kernelLog<<<16,16>>>(dest, operand, leftDesc);
    cudaDeviceSynchronize();
}

extern "C" void CCfusionOpForwardOp(float* dest, float* predictions, float* labels, TensorDesc* desc)
{
    _kernelCCfusionForward<<<16,16>>>(dest, predictions, labels, desc);
    cudaDeviceSynchronize();
}

extern "C" void CCfusionOpBackwardOp(float* dest, float* predictions, float* labels, TensorDesc* desc)
{
    _kernelCCfusionBackward<<<16,16>>>(dest, predictions, labels, desc);
    cudaDeviceSynchronize();
}