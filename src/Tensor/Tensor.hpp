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
class Tensor
{
public:
    Tensor(TensorShape dim = {}, TensorType dtype = TensorType::float32);
    Tensor(Tensor& src) = delete;
    DevicePointer* getTensorPointer();
    DevicePointer* getCudaDescriptorPointer();
    void setTensor_HostToDevice(void* data);
    void setTensor_DeviceToDevice(DevicePointer* data);
    void setTensor_DeviceToDevice(DevicePointer* data, unsigned int byteSize, unsigned int offset = 0);
    char* getTensorValues();
    void printTensor(FILE* stream, unsigned int print_max = 0);
    unsigned int getNumberOfElements();
    unsigned int getRank();
    TensorShape getShape();
    TensorType getType();
    void buildDescriptors();
    ~Tensor();

    // Tensor ops
    void tensorReshape(TensorShape newShape);
    static void addTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void subtractTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void mulTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void divideTensors(Tensor* dest, Tensor* left, Tensor* right);
    static void reduceTensor(cudnnReduceTensorDescriptor_t reduceDesc, Tensor* dest, Tensor* src);
    static void axisAlignedAccumulation(Tensor* dest, Tensor* src);
    static void matmul(Tensor* dest, Tensor* left, Tensor* right, bool transposeLeft = false, bool transposeRight= false);
    static void scaleByConstant(Tensor* dest, Tensor* operand, DevicePointer* scalar);
    static void activationForward(cudnnActivationDescriptor_t opDesc, Tensor* dest, Tensor* operand);
    static void activationBackward(cudnnActivationDescriptor_t opDesc, Tensor* dest, Tensor* grad, 
                                                                Tensor* prevOutput, Tensor* prevInput);
    static void softmaxForward(Tensor* dest, Tensor* operand);
    static void exp(Tensor* dest, Tensor* operand);
    static void log(Tensor* dest, Tensor* operand);
    static void CCfusionOpForward(Tensor* dest, Tensor* predictions, Tensor* labels);
    static void CCfusionOpBackward(Tensor* dest, Tensor* predictions, Tensor* labels);
    //Tensor helper
    static Tensor* createWithConstant(float value, TensorShape shape, TensorType dtype = TensorType::float32);
private:
    TensorShape shape;
    TensorType dtype;
    DevicePointer* tensorDeviceMemory;
    unsigned int rank;
    bool scalarTensor;
    cudnnTensorDescriptor_t cudnnTensorDescriptor;
    TensorDesc *cudaDescriptorDevice;
    bool cudnnDescriptorInitialized; 
};
#endif