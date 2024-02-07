/*
C/C++ api for tensor cuda operations. Supports tensors with rank up to 5


*/

#ifndef CUDA_KERNEL_LIB
#define CUDA_KERNEL_LIB

#ifdef __cplusplus 
extern "C"{ 
#endif

struct TensorDesc
{
    unsigned char ndim;
    unsigned char dim[5];
    unsigned int dimStrides[5];
};

void addTensors( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);

#ifdef __cplusplus 
}
#endif

#endif