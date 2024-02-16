#include "SGD.hpp"

SGD::SGD(float learningRate)
:
Optimizer()
{
    cudaMalloc(&deviceLearnigRate, sizeof(float));
    cudaMemcpy(deviceLearnigRate, &learningRate, sizeof(float), cudaMemcpyHostToDevice);
}

void SGD::build(std::vector<Variable *>& variables)
{
}

void SGD::updateGradient(const Variable *variable, Tensor *grad)
{
    Tensor::scaleByConstant(grad, grad, deviceLearnigRate);
}

SGD::~SGD()
{
    cudaFree(deviceLearnigRate);
}
