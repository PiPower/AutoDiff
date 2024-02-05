#include "Expression.hpp"

Expression::Expression()
:
visited(false), result(nullptr), addedToExecutionList(false)
{
    initCublas();
}

Expression::~Expression()
{
}