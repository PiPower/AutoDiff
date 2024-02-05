#include "Variable.hpp"
#include "../Utils/error_logs.hpp"


Variable::Variable(TensorShape shape, Initializer* initializer, TensorType dtype)
:
Expression(), initializer(initializer)
{
    logErrorAndExit(initializer == nullptr, "initializer cannot be nullptr \n");
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
    
}
