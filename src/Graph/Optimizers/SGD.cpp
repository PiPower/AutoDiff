#include "SGD.hpp"

SGD::SGD(float learningRate)
:
Optimizer()
{
    cudaError_t err;
    err = cudaMalloc(&deviceLearnigRate, sizeof(float));
    logErrorAndExit(err != cudaSuccess, "Could not alloc memory for learning rate");
    err = cudaMemcpy(deviceLearnigRate, &learningRate, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "Could not copy learning rate into device");
}

void SGD::build(std::vector<Variable *>& variables)
{
}

void SGD::updateGradient(const Variable *variable, Tensor *grad)
{
    Tensor::scaleByConstant(grad, grad, deviceLearnigRate);
}

void SGD::nextLoop()
{
}

SGD::~SGD()
{
    cudaFree(deviceLearnigRate);
}
