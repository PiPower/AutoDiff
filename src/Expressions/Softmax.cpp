#include "Softmax.hpp"

Softmax::Softmax(Expression *expr, std::vector<unsigned int> axis)
:
axis(axis)
{
    children.push_back(expr);
    opDescriptor = createCudnnReduceDescriptor(CUDNN_REDUCE_TENSOR_ADD);
}

void Softmax::build()
{
    // all axis to be reduced are set to 1
    reducedShape = children[0]->getTensor()->getShape();

    for(int i=0; i < axis.size(); i++)
    {
         logErrorAndExit(axis[i] >=children[0]->getTensor()->getRank(), "Axis larger than tensor dimensions allow in Softmax");
    }

    for(const unsigned int& dim : axis )
    {
        reducedShape[dim] = 1;
    }

    result = new Tensor(children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());
    grad_out_prod = new Tensor(children[0]->getTensor()->getShape(), children[0]->getTensor()->getType());
    intermidiate = new Tensor(reducedShape, children[0]->getTensor()->getType());
}

void Softmax::execute()
{
    Tensor::exp(result, children[0]->getTensor());
    Tensor::reduceTensor(opDescriptor, intermidiate, result);
    Tensor::divideTensors(result, result, intermidiate);
}

void Softmax::backwardPass(Tensor *propagatedGradient, BackwardData &storedGradients)
{
    // softmax grad for y_j = s_j(x_1, ..., x_n) is as follows
    // dL/dx_j = y_j * (dL/dy_j - sum_red( dL/dy * y)
    Tensor::mulTensors(grad_out_prod, propagatedGradient, result);
    Tensor *grad = new Tensor( children[0]->getTensor()->getShape(),  children[0]->getTensor()->getType());

    Tensor::reduceTensor(opDescriptor, intermidiate, grad_out_prod);
    Tensor::subtractTensors(grad_out_prod, propagatedGradient, intermidiate);
    Tensor::mulTensors(grad,grad_out_prod, result);
    
    storedGradients.gradientTensors.push_back(grad);
    storedGradients.nodeAddres.push_back(children[0]);
}
