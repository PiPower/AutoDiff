#include "./Expression.hpp"

#ifndef ACTIVATION 
#define ACTIVATION

enum class ActivationType
{
    sigmoid,
    relu,
    tanh,
    clipped_relu,
    elu,
    identity,
    swish
};

class Activation : public Expression
{
public:
    Activation(Expression* expr_in,ActivationType activation, double functionData = 1000000);
    virtual void build();
    virtual void execute();
    virtual void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    double functionData;
    ActivationType activation;
    cudnnActivationDescriptor_t  opDescriptor;
};

#endif