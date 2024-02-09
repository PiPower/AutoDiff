#include "Expression.hpp"

#ifndef REDUCE_MEAN
#define REDUCE_MEAN
/*
f = reduce(x, axis)
Calculates sum over given axis
*/
class ReduceMean: public Expression
{
public:
    ReduceMean( Expression* child_node, std::vector<unsigned int> reduce_axis, bool keepDim = false);
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    std::vector<unsigned int> axis;
    TensorShape newShape;
    bool keepDim;
    cudnnReduceTensorDescriptor_t  opDescriptor;
    Tensor* constant; 
    TensorShape reducedShape;
};

#endif