#include "Expression.hpp"

Expression::Expression()
:
visited(false), result(nullptr), addedToExecutionList(false)
{
    initCublas();
    initCudnn();
}

Expression::~Expression()
{
}