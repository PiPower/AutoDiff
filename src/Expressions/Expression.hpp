#include "../Tensor/Tensor.hpp"

#ifndef EXPRESSION
#define EXPRESSION

class Graph;
class Expression
{
public:
    virtual void compile() = 0;
    virtual ~Expression();
protected:
friend class Graph;
    bool visited;
    bool addedToExecutionList;
    std::vector<Expression*> children;
    Expression();
    Tensor* result;
};

#endif