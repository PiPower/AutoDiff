
add_library(Tensor STATIC
    Tensor.cpp
    CudaManagers/CublasManager.cpp
    CudaManagers/CudnnManager.cpp
)
