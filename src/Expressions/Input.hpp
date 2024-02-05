#include"Expression.hpp"

#ifndef INPUT
#define INPUT
/*
Special type of Expression node. It's purpose is to pass input tensor
into parent node
*/
class Input : public Expression
{
public:
    Input() = delete;
    Input(TensorShape shape);
    void build();
private:
TensorShape shape;
};

#endif