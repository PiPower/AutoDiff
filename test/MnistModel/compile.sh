g++ -g ConvMnist.cpp MnistDataset.cpp -L../../build/Graph -lGraph \
-L../../build/Expressions -lExpressions -L../../build/Tensor -lTensor -L../../build/Utils -lUtils \
-L../../build/Kernels -lKernels \
-L /usr/local/cuda-12.3/lib64 -I /usr/local/cuda-12.3/include -lcudart -lcublas \
-L /usr/lib/x86_64-linux-gnu/ -I /usr/include/ -lcudnn_ops_infer -lcudnn