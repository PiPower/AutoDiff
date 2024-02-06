#include <vector>

#ifndef TENSOR 
#define TENSOR

typedef std::vector<unsigned int> TensorShape;

enum class TensorType
{
    uint16 ,
    uint32,
    uint64,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
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