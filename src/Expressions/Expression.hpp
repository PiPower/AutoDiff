#include "../Tensor/Tensor.hpp"

#ifndef EXPRESSION
#define EXPRESSION

class Expression
{
public:
    virtual void compile() = 0;
    virtual ~Expression();
protected:
    Expression() = default;
};

#endif