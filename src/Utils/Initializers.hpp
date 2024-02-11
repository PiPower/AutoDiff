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

class ConstantInitializer : public Initializer
{
public:
    ConstantInitializer(float constant, TensorType dtype);
    char* generate(unsigned int count);
    void setTensorType(TensorType dtype);
    ~ConstantInitializer();
private:
    TensorType dtype;
    float constant;
};

class GlorotUniform : public Initializer
{
public:
    GlorotUniform(float fan_in, float fan_out, TensorType dtype);
    char* generate(unsigned int count);
    void setTensorType(TensorType dtype);
    ~GlorotUniform();
private:
    TensorType dtype;
    float fan_in;
    float fan_out;
};


class InMemory : public Initializer
{
public:
    InMemory(float* data);
    char* generate(unsigned int count);
    void setTensorType(TensorType dtype);
    ~InMemory();
private:
   float* data;
};
#endif