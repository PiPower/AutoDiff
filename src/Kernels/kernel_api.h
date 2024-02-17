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
    unsigned int dim[5];
    unsigned int dimStrides[5];
};

void addTensorsOp(float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);
void mulTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);
void divideTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);
void axisAlignedAccumulationOp( float* dest, float* src, TensorDesc* destDesc, TensorDesc* srcDesc);
void addConstantOp(float* dest, float* operand, float* scalar, TensorDesc* leftDesc);
void scaleByConstantOp(float* dest, float* operand, float* scalar, TensorDesc* leftDesc);
void divideByConstantOp(float* dest, float* operand, float* scalar, TensorDesc* leftDesc);
void sqrtOp(float* dest, float* operand, TensorDesc* leftDesc);
void subtractTensorsOp( float* dest, float* left, float* right, TensorDesc* leftDesc, TensorDesc* rightDesc);
void expOp(float* dest, float* operand, TensorDesc* leftDesc);
void logOp(float* dest, float* operand, TensorDesc* leftDesc);
void CCfusionOpForwardOp(float* dest, float* predictions, float* labels, TensorDesc* desc);
void CCfusionOpBackwardOp(float* dest, float* predictions, float* labels, TensorDesc* desc);

#ifdef __cplusplus 
}
#endif

#endif