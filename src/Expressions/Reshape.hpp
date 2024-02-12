#include "Expression.hpp"

#ifndef RESHAPE
#define RESHAPE
/*
f = reduce(x, axis)
Calculates sum over given axis
*/
class Reshape: public Expression
{
public:
    Reshape( Expression* child_node, TensorShape newShape);
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    TensorShape oldShape;
    TensorShape newShape;
};

#endif