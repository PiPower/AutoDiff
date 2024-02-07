#include "Addition.hpp"
#include <iostream>
#include "../Utils/error_logs.hpp"
using namespace std;

Addition::Addition(Expression *left_side, Expression *right_side)
:
Expression()
{
    logErrorAndExit(left_side == nullptr, "ERROR: left child of node [Addition] cannot be nullptr \n");
    logErrorAndExit(right_side == nullptr, "ERROR: right child of node [Addition] cannot be nullptr \n");

    children.push_back(left_side);
    children.push_back(right_side);

}

void Addition::build()
{
    const TensorShape left_shape = children[0]->getTensor()->getShape();
    const TensorShape right_shape = children[1]->getTensor()->getShape();

    logErrorAndExit( children[0]->getTensor()->getType() != children[1]->getTensor()->getType(), 
    "Non matching tensor types for addition node \n");

    logErrorAndExit( left_shape.size() != right_shape.size(),"Unmatching tensor shapes for addition node \n");
    for(int i =0;  i < left_shape.size(); i++ )
    {
          logErrorAndExit(left_shape[i] != right_shape[i], "Unmatching tensor shapes for addition node \n");
          logErrorAndExit(left_shape[i] == 0, "Tensor cannot bet 0 along any dimension");
    }

    result = new Tensor(left_shape,  children[0]->getTensor()->getType());
    result->setTensor_DeviceToDevice(children[0]->getTensor()->getTensorPointer());

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

void Addition::execute()
{
    /*
    float alpha = 1.0;
    float beta = 1.0;
    result->setTensor_DeviceToDevice(children[0]->getTensor()->getTensorPointer());
    addTensors(&alpha, children[1]->getDescriptor(),children[1]->getTensor(), 
    &beta, tensorDescriptor, result);
    */
   addTensors((float*)result->getTensorPointer(), (float*)children[0]->getTensor()->getTensorPointer(),
   (float*)children[1]->getTensor()->getTensorPointer(), children[0]->getCudaDescriptor(), 
   children[1]->getCudaDescriptor());
    
}
