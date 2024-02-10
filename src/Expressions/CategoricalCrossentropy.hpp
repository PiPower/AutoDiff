#include "Expression.hpp"

#ifndef CATEGORICAL_CROSSENTROPY
#define CATEGORICAL_CROSSENTROPY
/*
f = reduce(x, axis)
Calculates sum over given axis
*/
class CategoricalCrossentropy: public Expression
{
public:
    CategoricalCrossentropy( Expression* prob_node,  Expression* label_node, std::vector<unsigned int> reduce_axis);
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    std::vector<unsigned int> axis;
    TensorShape newShape;
    cudnnReduceTensorDescriptor_t  opDescriptor;
    TensorShape reducedShape;
    Tensor* buffer;
};

#endif