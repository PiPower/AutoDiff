#include "Expression.hpp"

#ifndef ADDITION
#define ADDITION

/*
f = x + y
Output of addition layer has the same shape as right node 
*/
class Addition : public Expression
{
public:
    Addition( Expression* left_side, Expression* right_side);
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
};

#endif