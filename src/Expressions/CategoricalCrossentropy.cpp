#include "CategoricalCrossentropy.hpp"
#include <algorithm>

CategoricalCrossentropy::CategoricalCrossentropy(Expression *prob_node,  Expression* label_node, std::vector<unsigned int> reduce_axis)
:
axis(reduce_axis)
{
    logErrorAndExit(prob_node == nullptr, "Probability node of [CategoricalCrossentropy] cannot be nullptr \n");
    logErrorAndExit(label_node == nullptr, "Label node of [CategoricalCrossentropy] cannot be nullptr \n");
    logErrorAndExit(reduce_axis.size() ==0, "Reduction operation needs axis to be able to work");
    opDescriptor = createCudnnReduceDescriptor(CUDNN_REDUCE_TENSOR_ADD);
    children.push_back(prob_node);
    children.push_back(label_node);
    opDescriptor = createCudnnReduceDescriptor(CUDNN_REDUCE_TENSOR_ADD);
}

void CategoricalCrossentropy::build()
{
    logErrorAndExit( children[0]->getTensor()->getShape() != children[1]->getTensor()->getShape(),
                         "Unequal shapes for probability and label node \n");

    reducedShape = children[0]->getTensor()->getShape();

    for(int i=0; i < axis.size(); i++)
    {
        logErrorAndExit(axis[i] >=children[0]->getTensor()->getRank(), "Axis larger than tensor dimensions allow in CategoricalCrossentropy");
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

    result = new Tensor(reducedShape, children[0]->getTensor()->getType());
    buffer = new Tensor(children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());
}

void CategoricalCrossentropy::execute()
{
    Tensor::CCfusionOpForward(buffer, children[0]->getTensor(), children[1]->getTensor());
    Tensor::reduceTensor(opDescriptor, result, buffer);
}

void CategoricalCrossentropy::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    Tensor::CCfusionOpBackward(buffer, children[0]->getTensor(),  children[1]->getTensor());
    Tensor* grad = new Tensor(buffer->getShape(), buffer->getType());
    Tensor::mulTensors(grad, buffer,  propagatedGradient);

    storedGradients.gradientTensors.push_back(grad);
    storedGradients.nodeAddres.push_back( children[0]);
}
