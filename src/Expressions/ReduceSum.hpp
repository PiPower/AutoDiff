#include "Expression.hpp"

#ifndef REDUCE_SUM
#define REDUCE_SUM
/*
f = reduce(x, axis)
Calculates sum over given axis
*/
class ReduceSum: public Expression
{
public:
    ReduceSum( Expression* child_node, std::vector<unsigned int> reduce_axis, bool keepDim = false);
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    std::vector<unsigned int> axis;
    TensorShape newShape;
    bool keepDim;
    cudnnReduceTensorDescriptor_t  opDescriptor;
    Tensor* ones; 
    TensorShape reducedShape;
};

#endif