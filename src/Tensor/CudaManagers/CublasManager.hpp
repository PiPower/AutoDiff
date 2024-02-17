#include <cublas_v2.h>
#include "../TensorTypes.hpp"

#ifndef CUBLAS_MANAGER
#define CUBLAS_MANAGER

void initCublas(cudaStream_t stream);
void destroyCublas();

void cublasMatmul( bool transLeft, bool transRight, int m, int n, int k,
                const float *alpha, const float *left, int lda,
                const float *right, int ldb,
                const float *beta, float *dest, int ldc);
#endif