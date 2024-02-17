#include <vector>
#include "../Kernels/kernel_api.h"
#include "CudaManagers/CublasManager.hpp"
#include "CudaManagers/CudnnManager.hpp"
#include "TensorTypes.hpp"
#include <cstdio>
#ifndef TENSOR 
#define TENSOR

/*
Support up to 5D tensors
*/

struct shapeInfo
{
    cudnnTensorDescriptor_t cudnnTensorDescriptor;
    TensorDesc* cudaDescriptorDevice;
    bool scalar;
    TensorShape shape;
    unsigned int rank;
};

class Tensor
{
public:
    Tensor(TensorShape dim = {}, TensorType dtype = TensorType::float32);
    Tensor(Tensor& src);
    DevicePointer* getTensorPointer();
    DevicePointer* getCudaDescriptorPointer();
    void setTensor_HostToDevice(void* data);
    void setTensor_DeviceToDevice(Tensor* data);
    void setTensor_DeviceToDevice(DevicePointer* data);
    void setTensor_DeviceToDevice(DevicePointer* data, unsigned int byteSize, unsigned int offset = 0);
    char* getTensorValues();
    void printTensor(FILE* stream, unsigned int print_max = 0);
    unsigned int getNumberOfElements();
    int tensorAddShape(TensorShape newShape);
    int tensorReshape(int shapeIndex);
    unsigned int getRank();
    TensorShape getShape();
    TensorType getType();
    void buildDescriptors(shapeInfo* newShapeInfo);
    static void streamSync();
    ~Tensor();

    // Tensor ops
    static void addTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void subtractTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void mulTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void divideTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void reduceTensor(cudnnReduceTensorDescriptor_t reduceDesc, Tensor* dest, Tensor* src);
    static void axisAlignedAccumulation(Tensor* dest, Tensor* src);
    static void matmul(Tensor* dest, Tensor* left, Tensor* right, bool transposeLeft = false, bool transposeRight= false);
    static void addConstant(Tensor* dest, Tensor* operand, DevicePointer* scalar);
    static void scaleByConstant(Tensor* dest, Tensor* operand, DevicePointer* scalar);
    static void divideByConstant(Tensor* dest, Tensor* operand, DevicePointer* scalar);
    static void sqrt(Tensor* dest, Tensor* operand);
    static void activationForward(cudnnActivationDescriptor_t opDesc, Tensor* dest, Tensor* operand);
    static void activationBackward(cudnnActivationDescriptor_t opDesc, Tensor* dest, Tensor* grad, 
                                                                Tensor* prevOutput, Tensor* prevInput);
    static void softmaxForward(Tensor* dest, Tensor* operand);
    static void exp(Tensor* dest, Tensor* operand);
    static void log(Tensor* dest, Tensor* operand);
    static void CCfusionOpForward(Tensor* dest, Tensor* predictions, Tensor* labels);
    static void CCfusionOpBackward(Tensor* dest, Tensor* predictions, Tensor* labels);
    static void Convolution2DForward(Tensor* dest,Tensor* kernel, Tensor* input, cudnnFilterDescriptor_t kernelDesc,
        cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workspaceSize );

    static void backwardConv2dData(cudnnFilterDescriptor_t kernelDesc, Tensor *kernel, Tensor *propGrad,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, Tensor *grad);

    static void backwardConv2dFilter(Tensor *input, Tensor *propGrad, cudnnConvolutionDescriptor_t convDesc,
         cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, cudnnFilterDescriptor_t gradDesc, Tensor *grad);

    static void Pool2DForward(Tensor* dest, Tensor* input, cudnnPoolingDescriptor_t poolingDesc);

    static void Pool2DBackward(Tensor *prevInput, Tensor *propGrad, Tensor* prevOutput, Tensor *grad, cudnnPoolingDescriptor_t poolingDesc);
    //miscellaneous
    static Tensor* createWithConstant(float value, TensorShape shape, TensorType dtype = TensorType::float32);
    static std::vector<int> get2DConvOutputDim(cudnnConvolutionDescriptor_t opDesc,
                                                    Tensor* x, cudnnFilterDescriptor_t filterDesc);

    static size_t getConvAlgoWorkspaceSize(Tensor* dest, Tensor* input, cudnnFilterDescriptor_t kernelDesc,
        cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo );

    static cudnnConvolutionFwdAlgo_t getConvAlgo(Tensor* dest, Tensor* input, 
                    cudnnFilterDescriptor_t kernelDesc, cudnnConvolutionDescriptor_t convDesc);

    static cudnnConvolutionBwdDataAlgo_t getConvBackwardDataAlgo(Tensor* propagatedGrad, Tensor* grad,
     cudnnFilterDescriptor_t kernelDesc, cudnnConvolutionDescriptor_t convDesc );

    static cudnnConvolutionBwdFilterAlgo_t getConvBackwardFilterAlgo(Tensor* propagatedGrad, Tensor* input,
     cudnnFilterDescriptor_t gradDesc, cudnnConvolutionDescriptor_t convDesc );

    static size_t getConvBackwardDataAlgoWorkspaceSize(cudnnFilterDescriptor_t kernelDesc, 
    Tensor* propagatedGradDesc, cudnnConvolutionDescriptor_t  opDesc,  Tensor* grad_xDesc, cudnnConvolutionBwdDataAlgo_t algo);

    static size_t getConvBackwardFilterAlgoWorkspaceSize(cudnnFilterDescriptor_t gradDesc, 
    Tensor* propagatedGradDesc, cudnnConvolutionDescriptor_t  opDesc,  Tensor* inputDesc, cudnnConvolutionBwdFilterAlgo_t algo);

    static std::vector<int> get2DPoolingOutputDim(cudnnPoolingDescriptor_t opDesc, Tensor* input);
private:
    TensorShape shape;
    TensorType dtype;
    DevicePointer* tensorDeviceMemory;
    unsigned int rank;
    bool scalarTensor;
    cudnnTensorDescriptor_t cudnnTensorDescriptor;
    std::vector<shapeInfo> allRegisteredShapes;
    TensorDesc *cudaDescriptorDevice;
    int currentShapeIndex;
};
#endif