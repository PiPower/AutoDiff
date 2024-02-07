#include "../Tensor/Tensor.hpp"
#include "CudaManagers/CublasManager.hpp"
#include "CudaManagers/CudnnManager.hpp"
#include "../Utils/error_logs.hpp"
#include "../Kernels/kernel_api.h"

#ifndef EXPRESSION
#define EXPRESSION

#define MAX_TENSOR_RANK 5

class Graph;
class Expression
{
public:
    virtual void build() = 0;
    virtual void execute() = 0;
    virtual ~Expression();  
    Tensor* getTensor(){return result;}
    void* getDescriptor(){ return tensorDescriptor;}
    TensorDesc* getCudaDescriptor(){return cudaDescriptorDevice;}
protected:
    Expression();
protected:
friend class Graph;
    bool visited; 
    bool addedToExecutionList; 
    std::vector<Expression*> children;
    Tensor* result;
    void* tensorDescriptor;
    TensorDesc *cudaDescriptorDevice;
};

#endif