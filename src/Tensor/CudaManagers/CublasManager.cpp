#include "CublasManager.hpp"
#include <cstdio>

cublasHandle_t* cublasHandle = nullptr;

void cublasExitOnError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "%s",msg);
        fprintf(stderr, "Error code: %d \n", (int)status);
        exit(-1);
    }
}

void initCublas()
{
    if(cublasHandle != nullptr)
    {
        return;
    }

    cublasHandle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    cublasStatus_t status = cublasCreate(cublasHandle);
    cublasExitOnError(status, "Cublas initialization failed! \n");
    
#ifdef DEBUG
    status = cublasLoggerConfigure(true, 0,0,"./cublasLogs.txt" );
    cublasExitOnError(status, "Cublas could not start logging! \n");
#endif
}

void destroyCublas()
{
    if(cublasHandle != nullptr)
    {
        cublasStatus_t status = cublasDestroy(*cublasHandle);
        cublasExitOnError(status, "Cublas destruction failed! \n");
        delete cublasHandle;
        cublasHandle = nullptr;
    }
}

void cublasMatmul( bool transLeft, bool transRight, int m, int n, int k,
                const float *alpha, const float *left, int lda,
                const float *right, int ldb,
                const float *beta, float *dest, int ldc)
{
    cublasOperation_t tr_left = transLeft? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t tr_right = transRight? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t  status;
    status = cublasSgemm(*cublasHandle, tr_left, tr_right, 
                        m, n, k, alpha, left, 
                        lda, right, ldb, beta, dest, ldc);
    cublasExitOnError(status, "cublas matmul error \n");

}
