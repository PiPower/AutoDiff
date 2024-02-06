#include "Variable.hpp"
#include "../Utils/error_logs.hpp"


Variable::Variable(TensorShape shape, Initializer* initializer, TensorType dtype)
:
Expression(), initializer(initializer)
{
    logErrorAndExit(shape.size() == 0, "scalar tensors are currently unsuported\n");
    logErrorAndExit(initializer == nullptr, "initializer cannot be nullptr \n");
    for(int i =0;  i < shape.size(); i++ )
    {
          logErrorAndExit(shape[i] == 0, "Tensor cannot bet 0 along any dimension");
    }
    result = new Tensor(shape, dtype);
    initializer->setTensorType(dtype);
}

void Variable::initVariable()
{
    char* copySource = initializer->generate(result->getNumberOfElements());
    result->setTensor_HostToDevice(copySource);
    delete[] copySource;
}

void Variable::build()
{
    tensorDescriptor = createTensorDescriptor(result->getType(), result->getShape());
}

void Variable::execute()
{
}
