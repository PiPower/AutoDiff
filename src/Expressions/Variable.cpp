#include "Variable.hpp"
#include "../Utils/error_logs.hpp"


Variable::Variable(TensorShape shape, Initializer* initializer, TensorType dtype)
:
Expression(), initializer(initializer)
{
    logErrorAndExit(shape.size() == 0, "scalar tensors are currently unsuported\n");
    logErrorAndExit(initializer == nullptr, "initializer cannot be nullptr \n");
    logErrorAndExit(shape.size() > 5, "tensors of rank higher than 5 are not supported \n");
    logErrorAndExit(dtype !=TensorType::float32, "unsuportted tensor type \n");
    for(int i =0;  i < shape.size(); i++ )
    {
          logErrorAndExit(shape[i] == 0, "Tensor cannot bet 0 along any dimension\n");
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
    TensorDesc cudaDescriptor;
    cudaDescriptor.ndim = result->getShape().size();
    unsigned int stride =1;
    for(int i = cudaDescriptor.ndim -1 ; i >=0; i--)
    {
        cudaDescriptor.dim[i] = result->getShape()[i];
        cudaDescriptor.dimStrides[i] = stride;
        stride *= cudaDescriptor.dim[i];
    }
    cudaError_t err;
    err = cudaMalloc(&cudaDescriptorDevice, sizeof(TensorDesc));
    logErrorAndExit(err != cudaSuccess, "Could not allocate memory for tensor descriptor\n");
    err =cudaMemcpy(cudaDescriptorDevice, &cudaDescriptor, sizeof(TensorDesc), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "Could not set tensor descriptor on gpu side\n");
}

void Variable::execute()
{
}
