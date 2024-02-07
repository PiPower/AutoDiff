#include "kernel_api.h"

/*
tensor dest MUST have the same dimensions as left and must be packed.
each dimension x of left must satisfy x = c_i*y where y is 
corresponding dimension of right and c_i > 0 is constant whole number for ith dimension
If ubove conditions are not met, function has undefined behaviour
*/

__global__ void _kernelAddTensors(float* dest, float* left, float* right, 
                            TensorDesc* leftDesc, TensorDesc* rightDesc, unsigned int* upper_memory_bound)
{
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadIndex < *upper_memory_bound)
    {
        unsigned int memoryLocation = threadIndex;
        unsigned int rightOffset = 0;
        for(int i = 0; i < leftDesc->ndim ; i++)
        {   
            //find i-th dim in left tensor
            unsigned int leftDim = memoryLocation/leftDesc->dimStrides[i];
            //change memory location in left(larger) tensor
            memoryLocation = memoryLocation - leftDim * leftDesc->dimStrides[i]; 
            //find aligned offset in right(smaller) tensor
            rightOffset+= (leftDim%rightDesc->dim[i] ) * rightDesc->dimStrides[i];
        }
        dest[threadIndex] = left[threadIndex] + right[rightOffset]; 

        threadIndex = threadIndex + blockDim.x * gridDim.x;
    }


}





extern "C" void addTensors( float* dest, float* left, float* right, 
        TensorDesc* leftDesc, TensorDesc* rightDesc, unsigned int* upper_memory_bound)
{
     _kernelAddTensors<<<16,16>>>(dest, left, right, leftDesc, rightDesc, upper_memory_bound);
}


