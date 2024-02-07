#include "Expression.hpp"

Expression::Expression()
:
visited(false), result(nullptr), addedToExecutionList(false), 
tensorDescriptor(nullptr), cudaDescriptorDevice(nullptr)
{
    initCublas();
    initCudnn();
}

Expression::~Expression()
{
}