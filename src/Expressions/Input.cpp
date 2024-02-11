#include "Input.hpp"

using namespace std;

Input::Input(TensorShape shape, string name, bool label, TensorType dtype)
:
Expression(), name(name), label(label)
{
    for(int i =0;  i < shape.size(); i++ )
    {
          logErrorAndExit(shape[i] == 0, "Tensor cannot bet 0 along any dimension");
    }
    result = new Tensor(shape, dtype);
    holder = result;
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
    logErrorAndExit(holder->getShape() != t->getShape(), "incorrect shapes for assignment in Input node");
    logErrorAndExit(holder->getType() != t->getType(), "incorrect data type for assignment in Input node");
    result = t;
}

void Input::backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients)
{
    return;
}
