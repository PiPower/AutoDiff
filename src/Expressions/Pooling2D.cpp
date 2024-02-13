#include "Pooling2D.hpp"

Pooling2D::Pooling2D(Expression *node, PoolingType poolMode, Vec2 windowSize, Vec2 stride, Vec2 padding)
:
stride(stride), windowSize(windowSize), padding(padding)
{
    children.push_back(node);
    switch (poolMode)
    {
    case PoolingType::mean:
        mode = CUDNN_POOLING_MAX;
        break;
    case PoolingType::max:
        mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
    default:
        logErrorAndExit(true, "Unsupported pooling operation");
    }
}

void Pooling2D::build()
{
    using namespace std;
    
    poolingDesc = create2DPoolingDesc(mode, windowSize.y, windowSize.x, padding.y, padding.x, stride.y, stride.x);

    vector<int> out_dims = Tensor::get2DPoolingOutputDim(poolingDesc, children[0]->getTensor());
    TensorShape tensorShape;
    for(int dim : out_dims)
    {
        tensorShape.push_back((unsigned int)dim);
    }   

    result = new Tensor(tensorShape, children[0]->getTensor()->getType());
}

void Pooling2D::execute()
{
    Tensor::Pool2DForward(result, children[0]->getTensor(), poolingDesc);
}

void Pooling2D::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    Tensor* grad = new Tensor(children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());
    Tensor::Pool2DBackward(children[0]->getTensor(), propagatedGradient, result, grad, poolingDesc);

    storedGradients.gradientTensors.push_back(grad);
    storedGradients.nodeAddres.push_back(children[0]);
}
