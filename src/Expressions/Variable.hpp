#include"Expression.hpp"

#ifndef VARIABLE
#define VARIABLE

class Variable : public Expression
{
public:
    Variable(Tensor* tensor);
    void compile();
private:
Tensor* parameter;
};

#endif