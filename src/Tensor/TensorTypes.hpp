#ifndef TENSOR_TYPES
#define TENSOR_TYPES
#include <vector>

class Tensor;

typedef std::vector<unsigned int> TensorShape;
typedef void DevicePointer;

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

#endif