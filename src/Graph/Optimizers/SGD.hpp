#include "Optimizer.hpp"
#include <unordered_map>

#ifndef SGD_OPTIMIZER
#define SGD_OPTIMIZER

class SGD : public Optimizer
{
public:
    SGD(float learningRate);
    void build(std::vector<Variable*>& variables);
    void updateGradient(const Variable* variable, Tensor* grad);
    ~SGD();
private:
    float* deviceLearnigRate;
};




#endif