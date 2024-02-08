#include"Expression.hpp"

#ifndef MATMUL
#define MATMUL
/*
performs matmul
first to dimensions are considered the rest is treated sas batches
*/
class Matmul : public Expression
{
public:
    Matmul() = delete;
    Matmul(Expression* left, Expression* right, bool tr_left = false, bool tr_right = false, TensorType dtype = TensorType::float32);
    void build();
    void execute();
    void backwardPass(Tensor* propagatedGradient, BackwardData& storedGradients);
private:
    bool tr_left;
    bool tr_right;
};

#endif