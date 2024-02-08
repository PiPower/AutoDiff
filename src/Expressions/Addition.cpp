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
          logErrorAndExit(left_shape[i] == 0 && right_shape[i] !=0, "Tensor cannot bet 0 along any dimension");
          float tensorDimProp = ((float)left_shape[i] )/right_shape[i] ;
          float flooDimProp = floor(tensorDimProp);
          logErrorAndExit(tensorDimProp != flooDimProp, "Unmatching tensor shapes for addition node \n");
    }

    result = new Tensor(left_shape,  children[0]->getTensor()->getType());
    result->setTensor_DeviceToDevice(children[0]->getTensor()->getTensorPointer());
}

void Addition::execute()
{
   Tensor::addTensors(result, children[0]->getTensor(), children[1]->getTensor());
}

void Addition::backwardPass(Tensor *propagatedGradient, BackwardData& storedGradients)
{
    Tensor *grad_x = new Tensor( result->getShape(),  result->getType());
    Tensor *grad_y = new Tensor( children[1]->getTensor()->getShape(), children[1]->getTensor()->getType());
    //because we assumed that x_dim_i >= y_dim_i due to chain rule in case of 
    // x and y having different dims we must accumulate grad along correct axies of y
    grad_x->setTensor_DeviceToDevice(propagatedGradient->getTensorPointer());

    if(grad_y->getShape() !=  grad_x->getShape())
    {
        Tensor::axisAlignedAccumulation(grad_y, propagatedGradient);
    }
    else
    {
        grad_y->setTensor_DeviceToDevice(propagatedGradient->getTensorPointer());
    }

    storedGradients.nodeAddres.push_back(children[0]);
    storedGradients.nodeAddres.push_back(children[1]);

    storedGradients.gradientTensors.push_back(grad_x);
    storedGradients.gradientTensors.push_back(grad_y);

}
