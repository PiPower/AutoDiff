#include "Expression.hpp"

#ifndef ADDITION
#define ADDITION

class Addition : public Expression
{
public:
    Addition( Expression* left_side, Expression* right_side);
    void compile();
private:
    Expression* left_node;
    Expression* right_node;
};

#endif