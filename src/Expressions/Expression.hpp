#include "../Tensor/Tensor.hpp"
#include "CudaManagers/CublasManager.hpp"
#include "CudaManagers/CudnnManager.hpp"

#ifndef EXPRESSION
#define EXPRESSION

class Graph;
class Expression
{
public:
    virtual void build() = 0;
    virtual ~Expression();  
    Tensor* getTensor(){return result;}
protected:
    Expression();
protected:
friend class Graph;
    bool visited; 
    bool addedToExecutionList; 
    std::vector<Expression*> children;
    Tensor* result;
    void* tensorDescriptor;
};

#endif