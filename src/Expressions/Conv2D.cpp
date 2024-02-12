#include "Conv2D.hpp"

using namespace std;


Conv2D::Conv2D(Expression *image, Expression *kernel, Vec2 stride, Vec2 padding, Vec2 dilation )
:
Expression(), stride(stride), padding(padding), dilation(dilation)
{
    children.push_back(image);
    children.push_back(kernel);
    opDesc = createConv2Ddescriptor(padding.y, padding.x, stride.y, stride.x, dilation.y, dilation.x);
}

void Conv2D::build()
{
    TensorShape kernelShape = children[1]->getTensor()->getShape();
    TensorShape imgShape = children[0]->getTensor()->getShape();
    logErrorAndExit(kernelShape.size() != 4, "Incorrect kernel shape");
    logErrorAndExit(imgShape.size() != 4, "Incorrect image shape");
    logErrorAndExit(kernelShape[1] != imgShape[1], "kernel input channels not equal to image channels");
    
    filterDesc = createConv2DFilterDesc(kernelShape[0], kernelShape[1], kernelShape[2], kernelShape[3]);
    vector<int> out_dims = Tensor::get2DConvOutputDim(opDesc, children[0]->getTensor(), filterDesc);
    TensorShape tensorShape;
    for(int dim : out_dims)
    {
        tensorShape.push_back((unsigned int)dim);
    }   

    result = new Tensor(tensorShape, children[0]->getTensor()->getType());

    algo = Tensor::getConvAlgo(result, children[0]->getTensor(), filterDesc, opDesc);
    workspaceSize = Tensor::getConvAlgoWorkspaceSize(result, children[1]->getTensor(), 
                children[0]->getTensor(), filterDesc, opDesc, algo);
    cudaError_t status;
    status = cudaMalloc(&deviceWorkSpacePtr, workspaceSize);
    logErrorAndExit(status != cudaSuccess, "Could not allocate memory for convolution");
}

void Conv2D::execute()
{
    Tensor::Convolution2DForward(result, children[1]->getTensor(), children[0]->getTensor(), filterDesc, 
    opDesc, algo, deviceWorkSpacePtr, workspaceSize);
}

void Conv2D::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
}
