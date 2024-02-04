#include"Expression.hpp"

#ifndef VARIABLE
#define VARIABLE
/*
Special type of Expression node. It's purpose is to pass parameter tensor
into parent node and potentialy store gradient for step

*/
class Variable : public Expression
{
public:
    Variable() = delete;
    Variable(TensorShape shape);
    void compile();
private:
TensorShape shape;
Tensor* parameter;
Tensor* gradient;
};

#endif