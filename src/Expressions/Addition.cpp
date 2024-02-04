#include "Addition.hpp"

Addition::Addition(Expression *left_side, Expression *right_side)
:
Expression()
{
    children.push_back(left_node);
    children.push_back(right_side);
}

void Addition::compile()
{
}
