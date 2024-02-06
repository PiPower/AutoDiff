#include <cudnn.h>
#include "../../Tensor/Tensor.hpp"

#ifndef CUDNN_MANAGER
#define CUDNN_MANAGER

void initCudnn();
void destroyCudnn();

void* createTensorDescriptor(TensorType dtype, TensorShape shape);
void addTensors(const void *alpha,
                const void* aDesc, const void *A,
                const void *beta,const void* cDesc, void *C);
#endif