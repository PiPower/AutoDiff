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
    Input(TensorShape shape, std::string name, bool label = false, TensorType dtype = TensorType::float32);
    void build();
    void execute();
    const std::string* getName();
    void setInput(Tensor* t);
    bool isLabel(){return label;}
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    std::string name;
    Tensor* holder;
    bool label;
};

#endif