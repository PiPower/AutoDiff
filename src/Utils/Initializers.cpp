#include "Initializers.hpp"
#include <random>
using namespace std;

Initializer::~Initializer()
{
}


GaussianInitializer::GaussianInitializer(double mean, double std_dev, TensorType dtype)
:
Initializer(), mean(mean), std_dev(std_dev), dtype(dtype)
{

}

char* GaussianInitializer::generate(unsigned int count)
{
    if(dtype == TensorType::float32)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::normal_distribution<float> normal_dist(mean, std_dev);
        float* dataBuffer =  new float[count];
        for(int i =0; i < count; i++)
        {
            dataBuffer[i] = normal_dist(rng);
        }
        return (char*)dataBuffer;
    }
    return nullptr;
}

void GaussianInitializer::setTensorType(TensorType dtype)
{
    this->dtype = dtype;
}

GaussianInitializer::~GaussianInitializer()
{
}

ConstantInitializer::ConstantInitializer(float constant, TensorType dtype)
:
Initializer(), dtype(dtype), constant(constant)
{
}

char *ConstantInitializer::generate(unsigned int count)
{
    if(dtype == TensorType::float32)
    {
        float* dataBuffer =  new float[count];
        for(int i =0; i < count; i++)
        {
            dataBuffer[i] = constant;
        }
        return (char*)dataBuffer;
    }
    return nullptr;
}

void ConstantInitializer::setTensorType(TensorType dtype)
{
    this->dtype = dtype;
}

ConstantInitializer::~ConstantInitializer()
{
}

GlorotUniform::GlorotUniform(float fan_in, float fan_out, TensorType dtype)
:
Initializer(), fan_in(fan_in), fan_out(fan_out), dtype(dtype)
{
}

char *GlorotUniform::generate(unsigned int count)
{
    if(dtype == TensorType::float32)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        float limit = sqrt(6.0f/(fan_in + fan_out));
        std::uniform_real_distribution<float> glorot(-limit, limit);
        float* dataBuffer =  new float[count];
        for(int i =0; i < count; i++)
        {
            dataBuffer[i] = glorot(rng);
        }
        return (char*)dataBuffer;
    }
    return nullptr;
}

void GlorotUniform::setTensorType(TensorType dtype)
{
    this->dtype = dtype;
}

GlorotUniform::~GlorotUniform()
{
}

InMemory::InMemory(float *data)
:data(data)
{
}

char *InMemory::generate(unsigned int count)
{
    return (char*)data;
}

void InMemory::setTensorType(TensorType dtype)
{
}

InMemory::~InMemory()
{
}
