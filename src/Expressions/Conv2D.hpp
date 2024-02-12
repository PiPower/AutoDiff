#include "Expression.hpp"


#ifndef CONV_2D
#define  CONV_2D


struct Vec2
{
    int y;
    int x;
};

/*
Input image dim -> (batch_size, channels, height, width)
Kernel dims -> (out_channels, channels, height width)
*/
class Conv2D : public Expression
{
public:
    Conv2D(Expression* image, Expression* kernel, Vec2 stride = {1,1}, Vec2 padding = {0,0}, Vec2 dilation = {1,1} );
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    cudnnConvolutionDescriptor_t opDesc;
    cudnnFilterDescriptor_t filterDesc;
    Vec2 stride;
    Vec2 padding;
    Vec2 dilation;
    cudnnConvolutionFwdAlgo_t algoForward;
    cudnnConvolutionBwdDataAlgo_t algoBackwardData;    
    cudnnConvolutionBwdFilterAlgo_t  algoBackwardKernel;
    size_t workspaceSize;
    void* deviceWorkSpacePtr;
};



#endif