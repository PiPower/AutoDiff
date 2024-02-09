#include "Activation.hpp"

Activation::Activation(Expression *expr_in, ActivationType activation, double functionData)
:
activation(activation),functionData(functionData)
{
    children.push_back(expr_in);
}

void Activation::build()
{
    cudnnActivationMode_t mode;
    switch (activation)
    {
    case ActivationType::sigmoid:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
    case ActivationType::relu:
        mode = CUDNN_ACTIVATION_RELU;
        break;
    case ActivationType::tanh:
        mode = CUDNN_ACTIVATION_TANH;
        break;
    case ActivationType::clipped_relu:
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
    case ActivationType::elu:
        mode = CUDNN_ACTIVATION_ELU;
        break;
    case ActivationType::identity:
        mode = CUDNN_ACTIVATION_IDENTITY;
        break;
    case ActivationType::swish:
        mode = CUDNN_ACTIVATION_SWISH;
        break;
    default:
        logErrorAndExit(true, "Unsupported activation function");
        break;
    }

    opDescriptor = createCudnnActivationDescriptor(mode, functionData);
    result = new Tensor(children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());

}

void Activation::execute()
{
    Tensor::activationForward(opDescriptor, result, children[0]->getTensor());
}

void Activation::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    Tensor* grad = new Tensor( children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());

    Tensor::activationBackward(opDescriptor, grad, propagatedGradient, result, children[0]->getTensor());

    storedGradients.gradientTensors.push_back(grad);
    storedGradients.nodeAddres.push_back(children[0]);
}
