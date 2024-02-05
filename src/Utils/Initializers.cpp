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
        float* dataBuffer = (float*) malloc(count * sizeof(float) );
        for(int i =0; i < count; i++)
        {
            dataBuffer[i] =  normal_dist(rng);
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


