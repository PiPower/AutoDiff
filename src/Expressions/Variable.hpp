#include "Expression.hpp"
#include "../Utils/Initializers.hpp"

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
    Variable(TensorShape shape, Initializer* initializer, TensorType dtype = TensorType::float32);
    void initVariable();
    void build();
private:
Initializer* initializer;
Tensor* gradient;
};

#endif