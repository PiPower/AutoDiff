#include "Adam.hpp"

enum class MemoryLayout
{
    epsilon = 0,
    beta_1 = 1,
    beta_2 = 2,
    one_minus_beta_1 = 3,
    one_minus_beta_2 = 4,
    beta_1_power = 5,
    beta_2_power = 6,
    one_minus_beta_1_power = 7,
    one_minus_beta_2_power = 8,
    learningRate = 9
};

Adam::Adam(float learningRate, float beta_1, float beta_2, float eps)
:
Optimizer(), beta_1(beta_1), beta_2(beta_2), loopCounter(0.0f)
{
    cudaError_t err;
    err = cudaMalloc(&device_opt_params, sizeof(float) * 10);
    logErrorAndExit(err != cudaSuccess, "could not allocate memory for optimizer params");
    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::epsilon, &eps, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy epsilon to optimizer params");

    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::beta_1, &beta_1, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy beta 1 to optimizer params");

    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::beta_2, &beta_2, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy beta 2 to optimizer params");

    float buffer = 1.0f - beta_1;
    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::one_minus_beta_1, &buffer, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy 1 minus beta 1 to optimizer params");

    buffer = 1.0f - beta_2;
    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::one_minus_beta_2, &buffer, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy 1 minus beta 2 to optimizer params");

    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::learningRate, &learningRate, sizeof(float), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy learning rate to optimizer params");
}

void Adam::build(std::vector<Variable *> &variables)
{
    for(Variable* var : variables)
    {
        g_t[var] = new Tensor(var->getTensor()->getShape(), var->getTensor()->getType());
        m_t[var] = Tensor::createWithConstant(0, var->getTensor()->getShape(), var->getTensor()->getType());
        v_t[var] = Tensor::createWithConstant(0, var->getTensor()->getShape(), var->getTensor()->getType());
        m_t_hat[var] = new Tensor(var->getTensor()->getShape(), var->getTensor()->getType());
        v_t_hat[var] = new Tensor(var->getTensor()->getShape(), var->getTensor()->getType());
    }
}

void Adam::updateGradient(const Variable *variable, Tensor *grad)
{   
    // m_t
    Tensor::scaleByConstant(g_t[variable], grad, device_opt_params + (int)MemoryLayout::one_minus_beta_1);
    Tensor::scaleByConstant(m_t[variable], m_t[variable], device_opt_params + (int)MemoryLayout::beta_1);
    Tensor::addTensors(m_t[variable], m_t[variable], g_t[variable]);
    // v_t 
    Tensor::mulTensors(g_t[variable], grad, grad);
    Tensor::scaleByConstant(g_t[variable], g_t[variable], device_opt_params + (int)MemoryLayout::one_minus_beta_2);
    Tensor::scaleByConstant(v_t[variable], v_t[variable], device_opt_params + (int)MemoryLayout::beta_2);
    Tensor::addTensors(v_t[variable], v_t[variable], g_t[variable]);
    //m_t_hat
    Tensor::divideByConstant(m_t_hat[variable], m_t[variable],  device_opt_params + (int)MemoryLayout::one_minus_beta_1_power);
    //v_t_hat
    Tensor::divideByConstant(v_t_hat[variable], v_t[variable],  device_opt_params + (int)MemoryLayout::one_minus_beta_2_power);
    //final grad
    Tensor::sqrt(grad, v_t_hat[variable]);
    Tensor::addConstant(grad, grad, device_opt_params + (int)MemoryLayout::epsilon);
    Tensor::divideTensors(grad, m_t_hat[variable], grad );
    //final scale by learning rate
    Tensor::scaleByConstant(grad, grad, device_opt_params + (int)MemoryLayout::learningRate);
}

void Adam::nextLoop()
{
    loopCounter++;
    float params[4] = {pow(beta_1, loopCounter), pow(beta_2, loopCounter), 
                1.0f - pow(beta_1, loopCounter), 1.0f - pow(beta_2, loopCounter)};

    cudaError_t err;
    err = cudaMemcpy(device_opt_params + (int)MemoryLayout::beta_1_power, params, sizeof(float) * 4, cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "could not copy params to device memory");
}
