#include "ReduceSum.hpp"
#include <algorithm>

using namespace std;
ReduceSum::ReduceSum(Expression *child_node, std::vector<unsigned int> reduce_axis, bool keepDim)
:
Expression(), keepDim(keepDim), axis(reduce_axis)
{
    logErrorAndExit(reduce_axis.size() ==0, "Reduction operation needs axis to be able to work");
    opDescriptor = createCudnnReduceDescriptor(CUDNN_REDUCE_TENSOR_ADD);
    children.push_back(child_node);
}

void ReduceSum::build()
{
    // all axis to be reduced are set to 1
    TensorShape reducedShape = children[0]->getTensor()->getShape();

    for(int i=0; i < reducedShape.size(); i++)
    {
        auto place = find(axis.begin(), axis.end(), i);
        if(place == axis.end()) newShape.push_back(reducedShape[i] );
    }

    for(const unsigned int& dim : axis )
    {
        reducedShape[dim] = 1;
    }

    result = new Tensor(reducedShape, children[0]->getTensor()->getType());
}

void ReduceSum::execute()
{
    Tensor::reduceTensor(opDescriptor, result,  children[0]->getTensor());
    if(!keepDim) result->tensorReshape(newShape);
}

BackwardData ReduceSum::backwardPass(Tensor *propagatetGradient)
{
    return BackwardData();
}