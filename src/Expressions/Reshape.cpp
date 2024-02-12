#include "Reshape.hpp"

Reshape::Reshape(Expression *child_node, TensorShape newShape)
:
newShape(newShape)
{
    children.push_back(child_node);
}

void Reshape::build()
{
    result = new Tensor(*children[0]->getTensor());
    result->tensorReshape(newShape);
    oldShape  = children[0]->getTensor()->getShape();
}

void Reshape::execute()
{
    result->setTensor_DeviceToDevice(children[0]->getTensor());
}

void Reshape::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    Tensor* grad = new Tensor(oldShape, result->getType());
    grad->setTensor_DeviceToDevice(propagatedGradient);

    storedGradients.gradientTensors.push_back(grad);
    storedGradients.nodeAddres.push_back(children[0]);
}
