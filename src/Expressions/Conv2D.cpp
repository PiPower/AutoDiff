#include "Conv2D.hpp"

using namespace std;


Conv2D::Conv2D(Expression *image, Expression *kernel, Vec2 stride, Vec2 padding, Vec2 dilation )
:
Expression(), stride(stride), padding(padding), dilation(dilation), deviceWorkSpacePtr(nullptr)
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

    algoForward = Tensor::getConvAlgo(result, children[0]->getTensor(), filterDesc, opDesc);
    // grad has the same desc as input(ie child 0) and propagated grad has the same desc as output of node() ie resul)
    algoBackwardData = Tensor::getConvBackwardDataAlgo(result, children[0]->getTensor(), filterDesc, opDesc);
    //grad desc has the same descriptor as filter
    algoBackwardKernel = Tensor::getConvBackwardFilterAlgo(result, children[0]->getTensor(), filterDesc, opDesc);


    size_t convForwardSize = Tensor::getConvAlgoWorkspaceSize(result, children[0]->getTensor(), filterDesc, opDesc, algoForward);
    size_t convBackwardDataSize =  Tensor::getConvBackwardDataAlgoWorkspaceSize(filterDesc, result, opDesc, children[0]->getTensor(), algoBackwardData);
    size_t convBackwardFilterSize =  Tensor::getConvBackwardFilterAlgoWorkspaceSize(filterDesc, result, opDesc, children[0]->getTensor(),algoBackwardKernel);

    workspaceSize = max(convForwardSize, convBackwardDataSize);
    workspaceSize = max(workspaceSize, convBackwardFilterSize);

    cudaError_t status;
    if(workspaceSize > 0 )
    {
        status = cudaMalloc(&deviceWorkSpacePtr, workspaceSize);
        logErrorAndExit(status != cudaSuccess, "Could not allocate memory for convolution");
    }

}

void Conv2D::execute()
{
    Tensor::Convolution2DForward(result, children[1]->getTensor(), children[0]->getTensor(), filterDesc, 
    opDesc, algoForward, deviceWorkSpacePtr, workspaceSize);
}

void Conv2D::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    Tensor* grad_input= new Tensor( children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());
    Tensor* grad_kernel = new Tensor( children[1]->getTensor()->getShape(), children[1]->getTensor()->getType());


    Tensor::backwardConv2dData(filterDesc, children[1]->getTensor(), propagatedGradient, 
                            opDesc, algoBackwardData, deviceWorkSpacePtr, workspaceSize, grad_input );

    Tensor::backwardConv2dFilter( children[0]->getTensor(), propagatedGradient, 
                opDesc,algoBackwardKernel, deviceWorkSpacePtr, workspaceSize, filterDesc, grad_kernel );

    storedGradients.gradientTensors.push_back(grad_input);
    storedGradients.nodeAddres.push_back(children[0]);

    storedGradients.gradientTensors.push_back(grad_kernel);
    storedGradients.nodeAddres.push_back(children[1]);
}
