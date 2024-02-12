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
void cudnnExitOnError(cudnnStatus_t status, const char* msg);

cudnnTensorDescriptor_t createCudnnDescriptor(TensorType dtype, TensorShape shape);
cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(cudnnReduceTensorOp_t reduce_op);
cudnnActivationDescriptor_t createCudnnActivationDescriptor(cudnnActivationMode_t mode, double coef);
cudnnConvolutionDescriptor_t  createConv2Ddescriptor(int padding_y, int padding_x, 
                                                    int stride_y, int stride_x,int dilation_y, int dilation_x );
cudnnFilterDescriptor_t createConv2DFilterDesc(int out, int in, int height, int width);

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
    
void softmaxFunctionForward(DevicePointer *dest,  DevicePointer *Operand, 
                         cudnnTensorDescriptor_t destDesc, cudnnTensorDescriptor_t  OperandDesc);

void cudnnConvolution2DForward(cudnnTensorDescriptor_t inputDesc,void *input,
                cudnnFilterDescriptor_t filterDesc, void *filter,cudnnConvolutionDescriptor_t convDesc,
                cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workspaceSize,
                cudnnTensorDescriptor_t resDesc, void *result );

size_t getConvolutionAlgoForwardSize( cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
                        cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo);

cudnnConvolutionFwdAlgo_t  findConvForwardAlgo(cudnnTensorDescriptor_t inputDesc, cudnnFilterDescriptor_t kernelDesc,
                            cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t destDesc);
#endif