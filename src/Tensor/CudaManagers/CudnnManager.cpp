#include "CudnnManager.hpp"
#include <cstdio>
#include <stdlib.h>
#include "../../Utils/error_logs.hpp"
#include <iostream>
cudnnHandle_t* cudnnHandle = nullptr;
DevicePointer* workSpaceDvcPointer = nullptr;
unsigned int workSpaceSize = 100000;

cudnnDataType_t getDataType(TensorType dtype)
{
    switch (dtype)
    {
        case TensorType::float16:
            return CUDNN_DATA_BFLOAT16;
        case TensorType::float32:
            return CUDNN_DATA_FLOAT;
        case TensorType::float64:
            return CUDNN_DATA_DOUBLE;
        case TensorType::int32:
            return CUDNN_DATA_INT32;
        case TensorType::int64:
            return CUDNN_DATA_INT64;
        default:
            fprintf(stderr, "Unsported dtype");
            exit(-1);
    }
}

int getDataTypeStride(TensorType dtype)
{
    switch (dtype)
    {
        case TensorType::float16:
            return 2;
        case TensorType::float32:
            return 4;
        case TensorType::float64:
            return 8;
        case TensorType::int32:
            return 4;
        case TensorType::int64:
            return 8;
        default:
            fprintf(stderr, "Unsported dtype");
            exit(-1);
    }
}


void cudnnExitOnError(cudnnStatus_t status, const char* msg)
{
    if(status != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s", msg);
        fprintf(stderr, "Error code: %d \n", (int)status);
        exit(-1);
    }
}

void initCudnn()
{
    if(cudnnHandle != nullptr)
    {
        return;
    }

    cudnnHandle = (cudnnHandle_t*)malloc(sizeof(cudnnHandle_t));
    cudnnStatus_t status = cudnnCreate(cudnnHandle);
    cudnnExitOnError(status, "Cudnn initialization failed! \n");
#ifdef LOG_CUDNN
//for logging to work CUDNN_LOGDEST_DBG MUST be set to desired output: stdout or stderr or file
    status = cudnnSetCallback(0x0F, NULL, NULL );
    cudnnExitOnError(status, "Cudnn could not start logging! \n");
#endif
    cudaError err;
    err = cudaMalloc(&workSpaceDvcPointer, workSpaceSize);
    logErrorAndExit(err != cudaSuccess, "could not allocate memory for cudnn workspace");
}

void destroyCudnn()
{
    if(cudnnHandle != nullptr)
    {
        cudnnStatus_t status = cudnnDestroy(*cudnnHandle);
        cudnnExitOnError(status, "Cudnn destruction failed! \n");
        delete cudnnHandle;
        cudnnHandle = nullptr;
        cudaFree(workSpaceDvcPointer);
    }
}

cudnnTensorDescriptor_t createCudnnDescriptor(TensorType dtype, TensorShape shape)
{
    cudnnTensorDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&desc);
    cudnnExitOnError(status, "Cudnn tensor descriptor failed! \n");

    int dimCount = shape.size() > 3 ? shape.size() : 4;
    int* dim = new int[dimCount];
    int* dimStride = new int[dimCount];

    int stride = 1;
    int z = shape.size();
    for(int i =  dimCount - 1, j = shape.size()-1; i >= 0; i--, j--)
    {
        dim[i] =  j >= 0? shape[j] : 1;
        dimStride[i] = stride;
        stride = stride * dim[i];
    } 

    status = cudnnSetTensorNdDescriptor(desc, getDataType(dtype), dimCount, dim, dimStride);
    //status = cudnnSetTensorNdDescriptorEx(desc, CUDNN_TENSOR_NHWC ,getDataType(dtype), dimCount, dim);
    cudnnExitOnError(status, "Cudnn tensor descriptor set failed! \n");
    delete[] dim;
    delete[] dimStride;

    return desc;
}

void destroyCudnnDescriptor(void *descriptor)
{
    cudnnStatus_t status;
    status = cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t )descriptor);
#ifdef DEBUG
    cudnnExitOnError(status, "Cudnn could not destroy descriptor \n");
#endif
}

void addTensors(const void *alpha,
                const void* OperandDesc,  DevicePointer* OperandDevice,
                const void *beta,const void* DestinationDesc, DevicePointer *DestinationDevice)
{
    cudnnStatus_t status;
    status = cudnnAddTensor(*cudnnHandle, alpha, (cudnnTensorDescriptor_t)OperandDesc,
    OperandDevice, beta, (cudnnTensorDescriptor_t)DestinationDesc, DestinationDevice);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudnnExitOnError(status, "Cudnn could not start logging! \n");
#endif
}

cudnnReduceTensorDescriptor_t createCudnnReduceDescriptor(cudnnReduceTensorOp_t reduce_op)
{
    cudnnReduceTensorDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateReduceTensorDescriptor(&desc); 
    cudnnExitOnError(status, "Cudnn tensor reduce descriptor failed! \n");
    status = cudnnSetReduceTensorDescriptor(desc, reduce_op, 
    CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES );

    return desc;
}

cudnnActivationDescriptor_t createCudnnActivationDescriptor(cudnnActivationMode_t mode, double coef)
{
    cudnnActivationDescriptor_t out;
    cudnnCreateActivationDescriptor(&out);
    cudnnStatus_t status = cudnnSetActivationDescriptor(out, mode, CUDNN_PROPAGATE_NAN, coef);
    cudnnExitOnError(status, "Cudnn activation descriptor failed! \n");

    return out;
}

cudnnConvolutionDescriptor_t createConv2Ddescriptor(int padding_y, int padding_x, 
                                                    int stride_y, int stride_x,int dilation_y, int dilation_x)
{
    cudnnConvolutionDescriptor_t descOut;
    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&descOut);
    cudnnExitOnError(status, "Cudnn convolution descriptor failed! \n");
    status = cudnnSetConvolution2dDescriptor(descOut, padding_y,padding_x, stride_y, 
                            stride_x, dilation_y, dilation_x, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnExitOnError(status, "Cudnn convolution setting descriptor failed! \n");
    return descOut;
}

cudnnFilterDescriptor_t createConv2DFilterDesc(int out, int in, int height, int width)
{
    cudnnFilterDescriptor_t descOut;
    cudnnStatus_t status = cudnnCreateFilterDescriptor(&descOut);
    cudnnExitOnError(status, "Cudnn convolution filter descriptor failed! \n");
    status = cudnnSetFilter4dDescriptor(descOut, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out, in, height, width);
    cudnnExitOnError(status, "Cudnn convolution filter setting failed! \n");

    return descOut;
}


void reduceTensors(const cudnnReduceTensorDescriptor_t reduceTensorDesc,  
                    const void *alpha, DevicePointer *Operand, const void* OperandDesc,
                    const void *beta, const void* DestinationDesc, DevicePointer *Destination)
{
    cudnnStatus_t status;
    status = cudnnReduceTensor(*cudnnHandle, reduceTensorDesc,nullptr,0,workSpaceDvcPointer,
    workSpaceSize,alpha,(cudnnTensorDescriptor_t)OperandDesc, Operand,
    beta, (cudnnTensorDescriptor_t)DestinationDesc, Destination);

    cudaDeviceSynchronize();
#ifdef DEBUG
    cudnnExitOnError(status, "reduce op error! \n");
#endif
}

void activationFunctionForward(cudnnActivationDescriptor_t opDesc, DevicePointer *dest, DevicePointer *src, 
cudnnTensorDescriptor_t destDesc, cudnnTensorDescriptor_t  srcDesc )
{
    float alpha = 1;
    float beta = 0;

    cudnnStatus_t status;
    status = cudnnActivationForward(*cudnnHandle, opDesc, &alpha, srcDesc, src,&beta, destDesc, dest);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudnnExitOnError(status, "activation function forward pass error! \n");
#endif
}

void activationFunctionBackward(cudnnActivationDescriptor_t opDesc, DevicePointer *dest, DevicePointer *grad,
 DevicePointer *prevOutput, DevicePointer *prevInput, cudnnTensorDescriptor_t destDesc, 
 cudnnTensorDescriptor_t gradDesc, cudnnTensorDescriptor_t prevOutputDesc, cudnnTensorDescriptor_t prevInputDesc)
{
    float alpha = 1;
    float beta = 0;

    cudnnStatus_t status;
    status = cudnnActivationBackward(*cudnnHandle, opDesc, &alpha, prevOutputDesc, prevOutput, gradDesc,
                                grad, prevInputDesc, prevInput, &beta, destDesc, dest );
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudnnExitOnError(status, "activation function backward pass error! \n");
#endif
}

void softmaxFunctionForward(DevicePointer *dest, DevicePointer *Operand, 
                        cudnnTensorDescriptor_t destDesc, cudnnTensorDescriptor_t OperandDesc)
{
    float alpha = 1;
    float beta = 0;

    cudnnStatus_t status;
    status = cudnnSoftmaxForward(*cudnnHandle,CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL ,
                                    &alpha, OperandDesc, Operand, &beta, destDesc, dest );
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudnnExitOnError(status, "softmax forward pass error! \n");
#endif
}


void cudnnConvolution2DForward(cudnnTensorDescriptor_t inputDesc,void *input,
                cudnnFilterDescriptor_t filterDesc, void *filter,cudnnConvolutionDescriptor_t convDesc,
                cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workspaceSize,
                cudnnTensorDescriptor_t resDesc,void *result
)
{
    float alpha = 1;
    float beta = 0;
    cudnnStatus_t status;

    status = cudnnConvolutionForward(*cudnnHandle, &alpha, inputDesc, input, filterDesc, 
                                    filter,convDesc, algo, workSpace, workspaceSize, &beta, resDesc, result );
    cudaDeviceSynchronize();

#ifdef DEBUG
    cudnnExitOnError(status, "convolution forward pass error! \n");
#endif
}

size_t getConvolutionAlgoForwardSize(cudnnTensorDescriptor_t inputDesc, cudnnFilterDescriptor_t kernelDesc, 
                    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t resultDesc, cudnnConvolutionFwdAlgo_t algo)
{
    size_t size;
    cudnnStatus_t status;
    status = cudnnGetConvolutionForwardWorkspaceSize(*cudnnHandle, inputDesc, kernelDesc, convDesc, resultDesc, algo, &size);
#ifdef DEBUG
    cudnnExitOnError(status, "get convolution algo size error! \n");
#endif
    return size;
}

cudnnConvolutionFwdAlgo_t findConvForwardAlgo(cudnnTensorDescriptor_t inputDesc, cudnnFilterDescriptor_t kernelDesc,
                                 cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t destDesc)
{
    int returnAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t  performanceArr[3];
    cudnnStatus_t status;
    status = cudnnFindConvolutionForwardAlgorithm(*cudnnHandle, inputDesc, kernelDesc, 
    convDesc, destDesc, 3, &returnAlgoCount,  performanceArr);
#ifdef DEBUG
    cudnnExitOnError(status, "get convolution algo size error! \n");
#endif
    return performanceArr[0].algo;
}
