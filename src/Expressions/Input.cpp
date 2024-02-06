#include "Input.hpp"


Input::Input(TensorShape shape, TensorType dtype)
:
Expression()
{
    
    for(int i =0;  i < shape.size(); i++ )
    {
          logErrorAndExit(shape[i] == 0, "Tensor cannot bet 0 along any dimension");
    }
    result = new Tensor(shape, dtype);
}

void Input::build()
{
    tensorDescriptor = createTensorDescriptor(result->getType(), result->getShape());
}

void Input::execute()
{
}
