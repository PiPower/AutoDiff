#include "../Tensor/Tensor.hpp"
#include "CudaManagers/CublasManager.hpp"

#ifndef EXPRESSION
#define EXPRESSION

class Graph;
class Expression
{
public:
    virtual void compile() = 0;
    virtual ~Expression();
protected:
    Expression();
protected:
friend class Graph;
    bool visited; 
    bool addedToExecutionList; 
    std::vector<Expression*> children;
    Tensor* result;
};

#endif