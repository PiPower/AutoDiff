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

class Tensor
{
public:
    Tensor(TensorShape dim = {}, TensorType dtype = TensorType::float32);
    void setTensor_HostToDevice(void* data);
    unsigned int getNumberOfElements();
private:
    TensorShape shape;
    TensorType dtype;
    void* tensorDeviceMemory;
};

#endif