#include <vector>

#ifndef TENSOR 
#define TENSOR

typedef std::vector<unsigned int> TensorShape;

enum class TensorType
{
    uint16 , //placeholder, currently unsupported
    uint32,//placeholder, currently unsupported
    uint64,//placeholder, currently unsupported
    int16,//placeholder, currently unsupported
    int32,//placeholder, currently unsupported
    int64,//placeholder, currently unsupported
    float16,//placeholder, currently unsupported
    float32,
    float64,//placeholder, currently unsupported
};

/*
Support up to 5D tensors
*/
class Tensor
{
public:
    Tensor(TensorShape dim = {}, TensorType dtype = TensorType::float32);
    void* getTensorPointer();
    void setTensor_HostToDevice(void* data);
    void setTensor_DeviceToDevice(void* data);
    char* getTensorValues();
    unsigned int getNumberOfElements();
    TensorShape getShape();
    TensorType getType();
private:
    TensorShape shape;
    TensorType dtype;
    void* tensorDeviceMemory;
    unsigned int rank;
};

#endif