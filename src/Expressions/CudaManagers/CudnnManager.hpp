#include <cudnn.h>
#include "../../Tensor/Tensor.hpp"

#ifndef CUDNN_MANAGER
#define CUDNN_MANAGER

void initCudnn();
void destroyCudnn();

cudnnTensorDescriptor_t createTensorDescriptor(TensorType dtype, TensorShape shape);
#endif