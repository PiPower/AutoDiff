#include "../Tensor/Tensor.hpp"

#ifndef INITALIZERS
#define INITALIZERS

class Initializer
{
public:
    virtual char* generate(unsigned int count) = 0;
    virtual void setTensorType(TensorType dtype) = 0;
    virtual ~Initializer();
protected:
   Initializer() = default;
};


class GaussianInitializer : public Initializer
{
public:
    GaussianInitializer(double mean, double std_dev, TensorType dtype);
    char* generate(unsigned int count);
    void setTensorType(TensorType dtype);
    ~GaussianInitializer();
private:
    TensorType dtype;
    double mean;
    double std_dev;
};

#endif