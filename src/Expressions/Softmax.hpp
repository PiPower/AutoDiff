#include "./Expression.hpp"

#ifndef SOFTMAX
#define SOFTMAX
//currently softmax is supported only along channel axis
//current implementation is very susceptible to extreme values,
//which may cause NANs 
class Softmax : public Expression
{
public:
    Softmax(Expression* expr, std::vector<unsigned int> axis);
    void build() ;
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    Tensor* intermidiate;
    Tensor* grad_out_prod;
    TensorShape reducedShape;
    std::vector<unsigned int> axis;
    cudnnReduceTensorDescriptor_t  opDescriptor;
};



#endif