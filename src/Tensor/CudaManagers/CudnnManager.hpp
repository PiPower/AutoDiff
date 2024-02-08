#include <cudnn.h>
#include "../TensorTypes.hpp"

#ifndef CUDNN_MANAGER
#define CUDNN_MANAGER


void initCudnn();
void destroyCudnn();

void* createCudnnDescriptor(TensorType dtype, TensorShape shape);
void destroyCudnnDescriptor(void* descriptor);

void addTensors(const void *alpha,
                const void* OperandDesc, DevicePointer *Operand,
                const void *beta,const void* DestinationDesc, DevicePointer *Destination);

void reduceTensors(const cudnnReduceTensorDescriptor_t reduceTensorDesc,  
                    const void *alpha, DevicePointer *Operand, const void* OperandDesc,
                    const void *beta, const void* DestinationDesc, DevicePointer *Destination);

cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(cudnnReduceTensorOp_t reduce_op);
#endif