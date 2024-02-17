#include "Optimizer.hpp"
#include <unordered_map>

#ifndef ADAM_OPTIMIZER
#define ADAM_OPTIMIZER
/*
Optimizer from 
Diederik P. Kingma, Jimmy Lei Ba  ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
https://arxiv.org/pdf/1412.6980.pdf
*/
class Adam : public Optimizer
{

public:
    Adam(float learningRate = 0.001, float beta_1 = 0.9, float beta_2 = 0.999, float eps = 0.0000001);
    void build(std::vector<Variable*>& variables);
    void updateGradient(const Variable* variable, Tensor* grad);
    void nextLoop();
    ~Adam();
private:
    float loopCounter;
    float beta_1;
    float beta_2;
    //device data
    float* device_opt_params;

    std::unordered_map<const Variable*, Tensor*> g_t;
    std::unordered_map<const Variable*, Tensor*> m_t;
    std::unordered_map<const Variable*, Tensor*> v_t;
    std::unordered_map<const Variable*, Tensor*> m_t_hat;
    std::unordered_map<const Variable*, Tensor*> v_t_hat;
};




#endif