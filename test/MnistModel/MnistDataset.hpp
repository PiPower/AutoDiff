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
    unsigned int getTrainBatchCount() {return trainBatchCount;};
    unsigned int getTestBatchCount() {return testBatchCount;};
    unsigned int getBatchSize() {return batchSize;};
    std::vector<Tensor*> getTrainBatch(unsigned int i);
    std::vector<Tensor*> getTestBatch(unsigned int i);
    void shuffle();
private:
    int parseNumber(char* dataset, int& startPos);
    void loadDataset(float* images, float* labels, char* fileData, int maxSize);
    void setTrainBatches();
    void setTestBatches();
private:
    char* trainDataDevice;
    char* trainLabesDevice;

    char* testDataDevice;
    char* testLabesDevice;

    unsigned int batchSize;

    unsigned int trainBatchCount;
    Tensor** trainBatchesImg;
    Tensor** trainBatchesLabes;
    std::vector<unsigned int> trainBatchAssignment;

    unsigned int testBatchCount;
    Tensor** testBatchesImg;
    Tensor** testBatchesLabes;
    std::vector<unsigned int> testBatchAssignment;
};



#endif