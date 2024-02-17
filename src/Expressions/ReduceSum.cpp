#include "ReduceSum.hpp"
#include <algorithm>

using namespace std;
ReduceSum::ReduceSum(Expression *child_node, std::vector<unsigned int> reduce_axis, bool keepDim)
:
Expression(), keepDim(keepDim), axis(reduce_axis)
{
    logErrorAndExit(child_node == nullptr, "Child node of [ReduceSum] cannot be nullptr \n");
    logErrorAndExit(reduce_axis.size() ==0, "Reduction operation needs axis to be able to work");
    opDescriptor = createCudnnReduceDescriptor(CUDNN_REDUCE_TENSOR_ADD);
    children.push_back(child_node);
}

void ReduceSum::build()
{
    // all axis to be reduced are set to 1
    reducedShape = children[0]->getTensor()->getShape();

    for(int i=0; i < axis.size(); i++)
    {
         logErrorAndExit(axis[i] >=children[0]->getTensor()->getRank(), "Axis larger than tensor dimensions allow in ReduceSum");
    }

    for(int i=0; i < reducedShape.size(); i++)
    {
        auto place = find(axis.begin(), axis.end(), i);
        if(place == axis.end()) newShape.push_back(reducedShape[i] );
    }

    for(const unsigned int& dim : axis )
    {
        reducedShape[dim] = 1;
    }

    //allocating tensor for backward pass 
    ones = Tensor::createWithConstant(1,children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());
    result = new Tensor(reducedShape, children[0]->getTensor()->getType());
}

void ReduceSum::execute()
{
    Tensor::reduceTensor(opDescriptor, result,  children[0]->getTensor());
    if(!keepDim)
    {
        Tensor::streamSync();
        result->tensorReshape(newShape);
    } 
}

void ReduceSum::backwardPass(Tensor *propagatedGradient, BackwardData& storedGradients)
{
    if(!keepDim)
    {
        Tensor::streamSync();
        propagatedGradient->tensorReshape(reducedShape);
    }
    Tensor *grad = new Tensor( children[0]->getTensor()->getShape(),  children[0]->getTensor()->getType());
    Tensor::mulTensors(grad, ones, propagatedGradient);

    storedGradients.nodeAddres.push_back(children[0]);
    storedGradients.gradientTensors.push_back(grad);
   
}
