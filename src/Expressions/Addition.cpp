#include "Addition.hpp"
#include <iostream>
using namespace std;

Addition::Addition(Expression *left_side, Expression *right_side)
:
Expression()
{
    if(left_side == nullptr)
    {
        cout << "ERROR: left child of node [Addition] cannot be nullptr \n";
        exit(-1); 
    }
    if(right_side == nullptr)
    {
        cout << "ERROR: right child of node [Addition] cannot be nullptr \n";
        exit(-1); 
    }
    children.push_back(left_side);
    children.push_back(right_side);
}

void Addition::compile()
{
}
