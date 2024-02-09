#include <cudnn.h>
#include "../TensorTypes.hpp"

#ifndef CUDNN_MANAGER
#define CUDNN_MANAGER

struct MatmulDesc
{
    cudnnBackendDescriptor_t executionPlan;
    cudnnBackendDescriptor_t varianPack;
};

void initCudnn();
void destroyCudnn();

cudnnTensorDescriptor_t createCudnnDescriptor(TensorType dtype, TensorShape shape);
cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(cudnnReduceTensorOp_t reduce_op);
cudnnActivationDescriptor_t createCudnnActivationDescriptor(cudnnActivationMode_t mode, double coef);

void destroyCudnnDescriptor(void* descriptor);

void addTensors(const void *alpha,
                const void* OperandDesc, DevicePointer *Operand,
                const void *beta,const void* DestinationDesc, DevicePointer *Destination);

void reduceTensors(const cudnnReduceTensorDescriptor_t reduceTensorDesc,  
                    const void *alpha, DevicePointer *Operand, const void* OperandDesc,
                    const void *beta, const void* DestinationDesc, DevicePointer *Destination);

void activationFunctionForward(cudnnActivationDescriptor_t opDesc, DevicePointer *dest, DevicePointer *src, 
                        cudnnTensorDescriptor_t destDesc, cudnnTensorDescriptor_t  srcDesc);

void activationFunctionBackward(cudnnActivationDescriptor_t opDesc, DevicePointer *dest, DevicePointer *grad, 
    DevicePointer* prevOutput, DevicePointer* prevInput, cudnnTensorDescriptor_t destDesc, cudnnTensorDescriptor_t  gradDesc,
    cudnnTensorDescriptor_t prevOutputDesc, cudnnTensorDescriptor_t  prevInputDesc);

#endif