#include "Input.hpp"

using namespace std;

Input::Input(TensorShape shape, string name,TensorType dtype)
:
Expression(), name(name)
{
    for(int i =0;  i < shape.size(); i++ )
    {
          logErrorAndExit(shape[i] == 0, "Tensor cannot bet 0 along any dimension");
    }
    result = new Tensor(shape, dtype);
}

void Input::build()
{

}

void Input::execute()
{
}

const string* Input::getName()
{
    return &name;
}

void Input::setInput(Tensor *t)
{
    logErrorAndExit(result->getShape() == t->getShape(), "incorrect shapes for assignment in Input node");
    result->setTensor_DeviceToDevice(t->getTensorPointer());
}

void Input::backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients)
{
    return;
}
