#include "Matmul.hpp"

using namespace std;

Matmul::Matmul(Expression* left, Expression* right, bool tr_left, bool tr_right, TensorType dtype)
:
tr_left(tr_left), tr_right(tr_right)
{
    children.push_back(left);
    children.push_back(right);
}

void Matmul::build()
{
    TensorShape leftShape = children[0]->getTensor()->getShape();
    TensorShape rightShape = children[1]->getTensor()->getShape();

    TensorShape resultShape;
    logErrorAndExit(children[0]->getTensor()->getType() != children[1]->getTensor()->getType(), 
                    "not matchind types betwen left and right tensor");
    logErrorAndExit(leftShape.size() != 2, "left child of matmul has too incorrect rank");

    if(tr_left) swap(leftShape[0], leftShape[1]);
    if(tr_right) swap(rightShape[0], rightShape[1]);

    logErrorAndExit(leftShape.size() != rightShape.size(), "right child of matmul has too incorrect rank");
    logErrorAndExit(leftShape[1] != rightShape[0], "incorrent left column/right row count");
    resultShape.push_back(leftShape[0] );
    resultShape.push_back(rightShape[1] );

    result = new Tensor(resultShape, children[0]->getTensor()->getType());
}

void Matmul::execute()
{
    Tensor::matmul(result, children[0]->getTensor(), children[1]->getTensor(), tr_left, tr_right);
}

void Matmul::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
}
