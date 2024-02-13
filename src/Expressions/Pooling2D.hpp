#include "Expression.hpp"

#ifndef POOLING2D
#define POOLING2D



enum class PoolingType
{
    mean,
    max
};

class Pooling2D : public Expression
{
public:
    Pooling2D(Expression* node, PoolingType poolMode, Vec2 windowSize = {2,2}, Vec2 stride = {2, 2}, Vec2 padding = {0,0});
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnPoolingMode_t mode;
    Vec2 windowSize; 
    Vec2 stride;
    Vec2 padding;
};



#endif