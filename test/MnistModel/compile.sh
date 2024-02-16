CUDA_LIB=/usr/local/cuda-12.3/lib64
CUDA_INLUDE=/usr/local/cuda-12.3/include
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDNN_INCLUDE=/usr/include/

g++ -g ConvMnist.cpp MnistDataset.cpp -L../../build/Graph -lGraph \
-L../../build/Expressions -lExpressions -L../../build/Tensor -lTensor -L../../build/Utils -lUtils \
-L../../build/Kernels -lKernels \
-L ${CUDA_LIB} -I ${CUDA_INLUDE} -lcudart -lcublas \
-L ${CUDNN_LIB} -I ${CUDNN_INCLUDE} -lcudnn_ops_infer -lcudnn