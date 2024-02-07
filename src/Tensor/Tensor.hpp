#include <vector>
#include "../Kernels/kernel_api.h"
#include "CudaManagers/CublasManager.hpp"
#include "CudaManagers/CudnnManager.hpp"
#include "TensorTypes.hpp"
#ifndef TENSOR 
#define TENSOR

/*
Support up to 5D tensors
*/
class Tensor
{
public:
    Tensor(TensorShape dim = {}, TensorType dtype = TensorType::float32);
    DevicePointer* getTensorPointer();
    DevicePointer* getCudaDescriptorPointer();
    void setTensor_HostToDevice(void* data);
    void setTensor_DeviceToDevice(void* data);
    char* getTensorValues();
    unsigned int getNumberOfElements();
    TensorShape getShape();
    TensorType getType();
    void buildDescriptors();
    ~Tensor();

    // Tensor ops
    static void addTensors(Tensor* dest, Tensor* left, Tensor* right);
private:
    TensorShape shape;
    TensorType dtype;
    DevicePointer* tensorDeviceMemory;
    unsigned int rank;
    DevicePointer* tensorDescriptor;
    TensorDesc *cudaDescriptorDevice;
};
#endif