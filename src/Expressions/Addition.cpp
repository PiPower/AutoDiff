#include "Addition.hpp"
#include <iostream>
#include "../Utils/error_logs.hpp"
using namespace std;

Addition::Addition(Expression *left_side, Expression *right_side)
:
Expression()
{
    logErrorAndExit(left_side == nullptr, "ERROR: left child of node [Addition] cannot be nullptr \n");
    logErrorAndExit(right_side == nullptr, "ERROR: right child of node [Addition] cannot be nullptr \n");

    children.push_back(left_side);
    children.push_back(right_side);
}

void Addition::compile()
{
}
