#include "Expression.hpp"

#ifndef ADDITION
#define ADDITION

class Addition : public Expression
{
public:
    Addition( Expression* left_side, Expression* right_side);
    void build();
};

#endif