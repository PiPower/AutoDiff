#include "Expression.hpp"

#ifndef REDUCE_SUM
#define REDUCE_SUM
/*
Calculates sum over given axis
*/
class ReduceSum: public Expression
{
public:
    ReduceSum( Expression* child_node, std::vector<unsigned  int> reduce_axis, bool keepDim = false);
    void build();
    void execute();
    BackwardData backwardPass(Tensor* propagatetGradient);
private:
    std::vector<unsigned int> axis;
    bool keepDim;
    cudnnReduceTensorDescriptor_t  opDescriptor;
};

#endif