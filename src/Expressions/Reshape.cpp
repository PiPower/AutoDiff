#include "Reshape.hpp"

Reshape::Reshape(Expression *child_node, TensorShape newShape)
:
newShape(newShape)
{
    children.push_back(child_node);
}

void Reshape::build()
{
    result = new Tensor(newShape, children[0]->getTensor()->getType());
    oldShape  = children[0]->getTensor()->getShape();
}

void Reshape::execute()
{
    //for future add streams and async memcpy
    result->setTensor_DeviceToDeviceAsync(children[0]->getTensor());
}

void Reshape::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    Tensor* grad = new Tensor(oldShape, result->getType());
    grad->setTensor_DeviceToDeviceAsync(propagatedGradient);

    storedGradients.gradientTensors.push_back(grad);
    storedGradients.nodeAddres.push_back(children[0]);
}
