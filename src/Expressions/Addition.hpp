#include "Expression.hpp"

class Addition : public Expression
{
public:
    Addition( Expression* left_side, Expression* right_side);
private:
    Expression* left_node;
    Expression* right_node;
};