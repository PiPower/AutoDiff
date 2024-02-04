#include "Expression.hpp"

class Addition : public Expression
{
public:
    Addition( Expression* left_side, Expression* right_side);
    void compile();
private:
    Expression* left_node;
    Expression* right_node;
    Tensor smb_tensor;
};