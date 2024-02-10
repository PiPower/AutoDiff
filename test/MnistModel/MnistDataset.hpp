#ifndef MNIST_DATASET
#define MNIST_DATASET
#include "../../src/Tensor/Tensor.hpp"
#include <string>
/*
    only supported format is csv:
    https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv
    char data is trasformed into float scaled to [0,1] and label into one hot vectors
*/
class MnistDataset
{
public:
    MnistDataset(const char* trainPath, const char* testPath);
    void print_image(float* values, float* labels);
    void buildBatches(unsigned int batch_size);
    unsigned int getBatchCount() {return batchCount;};
    std::vector<Tensor*> getBatch(unsigned int i);
private:
    int parseNumber(char* dataset, int& startPos);
    void loadDataset(float* images, float* labels, char* fileData, int maxSize);
private:
    float* trainDataDevice;
    float* trainLabesDevice;

    float* testDataDevice;
    float* testLabesDevice;

    unsigned int batchSize;
    unsigned int batchCount;
    Tensor** batches;
    std::vector<unsigned int> trainBatchAssignment;
    std::vector<unsigned int> testBatchAssignment;
};



#endif