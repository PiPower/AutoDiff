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

void Addition::build()
{
    const TensorShape left_shape = children[0]->getTensor()->getShape();
    const TensorShape right_shape = children[1]->getTensor()->getShape();

    logErrorAndExit( children[0]->getTensor()->getType() != children[1]->getTensor()->getType(), 
    "Non matching tensor types for addition node \n");

    logErrorAndExit( left_shape.size() != right_shape.size(),"Unmatching tensor shapes for addition node \n");
    for(int i =0;  i < left_shape.size(); i++ )
    {
          logErrorAndExit(left_shape[i] != right_shape[i], "Unmatching tensor shapes for addition node \n");
          logErrorAndExit(left_shape[i] == 0, "Tensor cannot bet 0 along any dimension");
    }

    result = new Tensor(left_shape,  children[0]->getTensor()->getType());
    result->setTensor_DeviceToDevice(children[0]->getTensor()->getTensorPointer());
}

void Addition::execute()
{
   Tensor::addTensors(result, children[0]->getTensor(), children[1]->getTensor());
    
}
