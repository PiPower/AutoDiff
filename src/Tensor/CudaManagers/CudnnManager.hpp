#include <cudnn.h>
#include "../TensorTypes.hpp"

#ifndef CUDNN_MANAGER
#define CUDNN_MANAGER


void initCudnn();
void destroyCudnn();

void* createTensorDescriptor(TensorType dtype, TensorShape shape);
void addTensors(const void *alpha,
                const void* aDesc, DevicePointer *Operand,
                const void *beta,const void* cDesc, DevicePointer *Destination);
#endif