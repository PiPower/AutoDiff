#include "Tensor.hpp"
#include <cuda_runtime_api.h>
#include "../Utils/error_logs.hpp"

unsigned char typeSizeTable[] = {2, 4, 8, 2, 4, 8, 2, 4, 8};


Tensor::Tensor(TensorShape dim, TensorType dtype)
:
tensorDeviceMemory(nullptr), dtype(dtype), cudnnDescriptorInitialized(false)
{
    initCublas();
    initCudnn();

    logErrorAndExit(dtype != TensorType::float32, "currently usupported tensor type\n");
    this->shape = dim;
    rank = dim.size();
    cudaError err;
    if(rank != 0 )
    {
        scalarTensor = false;
        unsigned int total_item_count = 1;
        for(auto& dimSize : dim)
        {
           logErrorAndExit(dimSize == 0, "Tensor cannot have dim of size 0! \n");
           total_item_count = dimSize * total_item_count;
        }

        err = cudaMalloc(&tensorDeviceMemory, total_item_count * typeSizeTable[(unsigned int)dtype]);
    }
    else
    {
        // rank 0 tensor ie scalar
        scalarTensor = true;
        err =cudaMalloc(&tensorDeviceMemory, typeSizeTable[(unsigned int)dtype]);
    }
    logErrorAndExit(err != cudaSuccess, "Could not allocate memory for tensor on GPU");

    err = cudaMalloc((void**)&cudaDescriptorDevice, sizeof(TensorDesc));
    logErrorAndExit(err != cudaSuccess, "Could not allocate memory for tensor descriptor\n");
    buildDescriptors();
}

Tensor::Tensor(Tensor& src)
:
Tensor(src.shape, src.dtype)
{
    setTensor_DeviceToDevice(src.tensorDeviceMemory);
}

void Tensor::setTensor_HostToDevice(void* data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    logErrorAndExit(data == nullptr, "Copy source is unallocated tensor!\n");
    cudaError err;
    err = cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
}

void* Tensor::getTensorPointer()
{
    return tensorDeviceMemory;
}

DevicePointer *Tensor::getCudaDescriptorPointer()
{
    return cudaDescriptorDevice;
}

unsigned int Tensor::getNumberOfElements()
{
    if(scalarTensor) return 1;

    unsigned int total_size = 1;
    for(auto& dimSize : shape)
    {
        total_size *= dimSize;
    }

    return total_size;
}

unsigned int Tensor::getRank()
{
    return rank;
}

void Tensor::setTensor_DeviceToDevice(void *data)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    logErrorAndExit(data == nullptr, "Copy source is unallocated tensor!\n");
    cudaError err;
    err = cudaMemcpy(tensorDeviceMemory, data,  getNumberOfElements() * typeSizeTable[(unsigned int)dtype], cudaMemcpyDeviceToDevice );
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
}

TensorShape Tensor::getShape()
{
    return shape;
}

TensorType Tensor::getType()
{
    return dtype;
}

void Tensor::setTensor_DeviceToDevice(DevicePointer *data, unsigned int byteSize, unsigned int offset)
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    logErrorAndExit(data == nullptr, "Copy source is unallocated tensor!\n");
    cudaError err;
    err = cudaMemcpy((char*)tensorDeviceMemory + offset, data,  byteSize, cudaMemcpyDeviceToDevice );
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
}

char *Tensor::getTensorValues()
{
    logErrorAndExit(tensorDeviceMemory == nullptr, "Copy dest is unallocated  tensor!\n");
    char* data = new char[getNumberOfElements() * typeSizeTable[(unsigned int)dtype]];
    unsigned int tensor_byte_size = getNumberOfElements() * typeSizeTable[(unsigned int)dtype];
    cudaError err;
    err = cudaMemcpy(data, tensorDeviceMemory,  tensor_byte_size, cudaMemcpyDeviceToHost);
    logErrorAndExit(err != cudaSuccess, "Incorrent memory device to device copy");
    return data;
}

void Tensor::printTensor(FILE* stream, unsigned int print_max)
{
    float* data = (float*) getTensorValues();
    if(rank == 0)
    {
        fprintf(stream, "%.5f \n\n",*data );
        fflush(stream);
        return;
    }
    for(int i= 0; i < getNumberOfElements(); i++)
    {
        fprintf(stream, "%.5f ",data[i] );
        if((i+1) %shape[rank-1] == 0 )
            fprintf(stream, "\n");
        if(print_max > 0 && i >= print_max)
            break;
    }
    fprintf(stream, "\n");
    fflush(stream);
    delete[] data;

}

void Tensor::buildDescriptors()
{   
    if(cudnnDescriptorInitialized)
        destroyCudnnDescriptor(cudnnTensorDescriptor);

    TensorDesc cudaDescriptor;
    cudaDescriptor.ndim = rank;
    unsigned int stride =1;
    for(int i = cudaDescriptor.ndim -1 ; i >=0; i--)
    {
        cudaDescriptor.dim[i] = shape[i];
        cudaDescriptor.dimStrides[i] = stride;
        stride *= cudaDescriptor.dim[i];
    }
    cudaError err;
    err =cudaMemcpy(cudaDescriptorDevice, &cudaDescriptor, sizeof(TensorDesc), cudaMemcpyHostToDevice);
    logErrorAndExit(err != cudaSuccess, "Could not set tensor descriptor on gpu side\n");

    cudnnTensorDescriptor = createCudnnDescriptor(dtype, shape);
    cudnnDescriptorInitialized = true;
}

Tensor::~Tensor()
{
    cudaFree(tensorDeviceMemory);
    cudaFree(cudaDescriptorDevice);
    if(cudnnDescriptorInitialized)
        destroyCudnnDescriptor(cudnnTensorDescriptor);

}

void Tensor::addTensors(Tensor *dest, Tensor *left, Tensor *right)
{
    addTensorsOp((float*) dest->tensorDeviceMemory, (float*)left->tensorDeviceMemory, 
        (float*)right->tensorDeviceMemory, left->cudaDescriptorDevice, right->cudaDescriptorDevice);
}

void Tensor::subtractTensors(Tensor *dest, Tensor *left, Tensor *right)
{
    subtractTensorsOp((float*) dest->tensorDeviceMemory, (float*)left->tensorDeviceMemory, 
        (float*)right->tensorDeviceMemory, left->cudaDescriptorDevice, right->cudaDescriptorDevice);
}

void Tensor::mulTensors(Tensor *dest, Tensor *left, Tensor *right)
{
    mulTensorsOp((float*) dest->tensorDeviceMemory, (float*)left->tensorDeviceMemory, 
        (float*)right->tensorDeviceMemory, left->cudaDescriptorDevice, right->cudaDescriptorDevice);
}

void Tensor::divideTensors(Tensor *dest, Tensor *left, Tensor *right)
{
    divideTensorsOp((float*) dest->tensorDeviceMemory, (float*)left->tensorDeviceMemory, 
        (float*)right->tensorDeviceMemory, left->cudaDescriptorDevice, right->cudaDescriptorDevice);
}

void Tensor::reduceTensor(cudnnReduceTensorDescriptor_t reduceDesc, Tensor* dest, Tensor* src)
{
    float alpha = 1;
    float beta = 0;
    reduceTensors(reduceDesc,&alpha, src->tensorDeviceMemory, 
        src->cudnnTensorDescriptor, &beta, dest->cudnnTensorDescriptor, dest->tensorDeviceMemory);
}

void Tensor::axisAlignedAccumulation(Tensor *dest, Tensor *src)
{
    axisAlignedAccumulationOp((float*)dest->tensorDeviceMemory, (float*)src->tensorDeviceMemory,
    dest->cudaDescriptorDevice, src->cudaDescriptorDevice);
}

/*
    matmul of rank 2 tensors 
    if tensor rank!=2 behaviour is undefined
    We store tensors in row major format so in cublas instead of  A*B we get A^T*B^T
    we wish to find C^T
    from matmul properties for C = A * B we get C^T = B^T * A^T = A' *B' 
    ie reverse order of matricies
*/
void Tensor::matmul(Tensor *dest, Tensor *left, Tensor *right, bool transposeLeft, bool transposeRight)
{
    // negate because cublas expect column major
    std::swap(left, right);
    std::swap(transposeLeft, transposeRight);

    int a_rows = left->shape[1];
    int a_columns = left->shape[0];
    if(transposeLeft) std::swap(a_rows, a_columns);

    int b_rows = right->shape[1];
    int b_columns = right->shape[0];
    if(transposeRight) std::swap(b_rows, b_columns);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasMatmul(transposeLeft, transposeRight, a_rows, b_columns, a_columns, &alpha, (float*)left->tensorDeviceMemory,
    transposeLeft ? a_columns :a_rows, (float*)right->tensorDeviceMemory, transposeRight ? b_columns : a_columns,
    &beta, (float*)dest->tensorDeviceMemory, a_rows);
}

void Tensor::scaleByConstant(Tensor *dest, Tensor *operand, DevicePointer *scalar)
{
    logErrorAndExit(dest->getShape() != operand->getShape(), 
         "Not matching dimensions of dest and operand for scale by constant \n");
    scaleByConstantOp((float*)dest->tensorDeviceMemory, (float*)operand->tensorDeviceMemory,  (float*)scalar, operand->cudaDescriptorDevice);
}

void Tensor::activationForward(cudnnActivationDescriptor_t opDesc, Tensor *dest, Tensor *operand)
{
    activationFunctionForward(opDesc, dest->tensorDeviceMemory, operand->tensorDeviceMemory, 
    dest->cudnnTensorDescriptor, operand->cudnnTensorDescriptor);
}

void Tensor::activationBackward(cudnnActivationDescriptor_t opDesc, Tensor *dest, Tensor *grad, Tensor *prevOutput, Tensor *prevInput)
{
    activationFunctionBackward(opDesc, dest->tensorDeviceMemory, grad->tensorDeviceMemory, prevOutput->tensorDeviceMemory,
                                  prevInput->tensorDeviceMemory, dest->cudnnTensorDescriptor,grad->cudnnTensorDescriptor,
                                     prevOutput->cudnnTensorDescriptor, prevInput->cudnnTensorDescriptor);
}

void Tensor::softmaxForward(Tensor *dest, Tensor *operand)
{
    softmaxFunctionForward(dest->tensorDeviceMemory, operand->tensorDeviceMemory, 
                            dest->cudnnTensorDescriptor, operand->cudnnTensorDescriptor);
}

void Tensor::exp(Tensor *dest, Tensor *operand)
{
    expOp((float*)dest->tensorDeviceMemory,(float*) operand->tensorDeviceMemory, operand->cudaDescriptorDevice);
}

void Tensor::log(Tensor *dest, Tensor *operand)
{
    logOp((float*)dest->tensorDeviceMemory,(float*) operand->tensorDeviceMemory, operand->cudaDescriptorDevice);
}

void Tensor::CCfusionOpForward(Tensor *dest, Tensor *predictions, Tensor *labels)
{
    CCfusionOpForwardOp((float*)dest->tensorDeviceMemory, (float*)predictions->tensorDeviceMemory, 
                            (float*)labels->tensorDeviceMemory, predictions->cudaDescriptorDevice);
}

void Tensor::CCfusionOpBackward(Tensor *dest, Tensor *predictions, Tensor *labels)
{
     CCfusionOpBackwardOp((float*)dest->tensorDeviceMemory, (float*)predictions->tensorDeviceMemory, 
                            (float*)labels->tensorDeviceMemory, predictions->cudaDescriptorDevice);
}

void Tensor::tensorReshape(TensorShape newShape)
{
    unsigned int newNumberOfElements = 1;
    for(int i = newShape.size() -1 ; i >=0; i--)
    {
       newNumberOfElements*= newShape[i];
    }

    logErrorAndExit(newNumberOfElements != getNumberOfElements(), 
         "Not matching previous number of elements with new one \n");

    if(newShape.size() == 0 ) scalarTensor = true;
    if(newShape.size() > 0 ) scalarTensor = false;
    shape = newShape;
    rank = newShape.size();
    buildDescriptors();
}

Tensor *Tensor::createWithConstant(float value, TensorShape shape, TensorType dtype)
{
    Tensor* out = new Tensor(shape, dtype);
    float* mem = new float[out->getNumberOfElements()];
    for(int i=0; i < out->getNumberOfElements(); i++)
    {
        mem[i] = value;
    }
    out->setTensor_HostToDevice(mem);

    delete[] mem;
    return out;
}   
