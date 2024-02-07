/*
C/C++ api for tensor cuda operations. Supports tensors with rank up to 5


*/



#ifdef __cplusplus 
extern "C"{ 
#endif

struct TensorDesc
{
    unsigned char ndim;
    unsigned char dim[5];
    unsigned int dimStrides[5];
};

void addTensors( float* dest, float* left, float* right, 
        TensorDesc* leftDesc, TensorDesc* rightDesc, unsigned int* upper_memory_bound);

#ifdef __cplusplus 
}
#endif