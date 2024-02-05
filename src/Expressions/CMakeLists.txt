
add_library(Expressions STATIC
    Addition.cpp
    Expression.cpp
    Variable.cpp
    Input.cpp
    CudaManagers/CublasManager.cpp
    CudaManagers/CudnnManager.cpp
)