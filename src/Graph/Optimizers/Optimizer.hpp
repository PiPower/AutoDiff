#include "../../Expressions/Variable.hpp"
#include <vector>

#ifndef OPTIMIZER
#define OPTIMIZER

class Optimizer 
{
public:
    virtual void build(std::vector<Variable*>& variables) = 0;
    //updates gradient, variable is used a key for matching grad states
    virtual void updateGradient(const Variable* variable, Tensor* grad) = 0;
    virtual void nextLoop() = 0;
    ~Optimizer() {};
};

#endif