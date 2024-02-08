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

void addTensorsOp(float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);
void mulTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);
void axisAlignedAccumulationOp( float* dest, float* src, TensorDesc* destDesc, TensorDesc* srcDesc);

#ifdef __cplusplus 
}
#endif

#endif