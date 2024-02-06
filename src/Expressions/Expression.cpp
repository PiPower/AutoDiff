#include "Expression.hpp"

Expression::Expression()
:
visited(false), result(nullptr), addedToExecutionList(false), tensorDescriptor(nullptr)
{
    initCublas();
    initCudnn();
}

Expression::~Expression()
{
}