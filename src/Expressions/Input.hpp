#include"Expression.hpp"
#include <string>

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
    Input(TensorShape shape, std::string name, TensorType dtype = TensorType::float32);
    void build();
    void execute();
    const string* getName();
    void setInput(Tensor* t);
private:
    std::string name;
};

#endif