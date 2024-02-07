#include "../Tensor/Tensor.hpp"
#include "../Utils/error_logs.hpp"

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