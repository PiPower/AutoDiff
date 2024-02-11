#include "MnistDataset.hpp"
#include <fstream>
#include <iostream>

#define TRAIN_DATA_SIZE 60000
#define TEST_DATA_SIZE 10000
#define WIDTH 28
#define HEIGHT 28


using namespace std;



MnistDataset::MnistDataset(const char* trainPath, const char* testPath)
:
trainBatchesLabes(nullptr), trainBatchesImg(nullptr), 
testBatchesLabes(nullptr), testBatchesImg(nullptr)
{
    float* training_dataset_buffer = new float[28*28*60000];
    float* training_dataset_labels = new float[60000 * 10];

    ifstream file;
    file.open(trainPath,  std::ifstream::ate | std::ifstream::binary);
    if(!file.is_open())
    {
        cout << "could not open mnist train csv \n";
        exit(-1);
    }

    int size = file.tellg();
    char* dataset = new char[size];
    file.seekg(ios_base::beg);
    file.read(dataset, size);
    file.close();

    loadDataset(training_dataset_buffer, training_dataset_labels, dataset, size);

    cudaError_t err;
    err = cudaMalloc(&trainDataDevice, WIDTH * HEIGHT*TRAIN_DATA_SIZE * sizeof(float));
    if(err != cudaSuccess)
    {   
        cout<<"Could not allocate data for train images \n";
        exit(-1);
    }
    err = cudaMemcpy(trainDataDevice, training_dataset_buffer,  WIDTH * HEIGHT*TRAIN_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {   
        cout<<"Could not copy train image data to device \n";
        exit(-1);
    }

    err = cudaMalloc(&trainLabesDevice,10 * TRAIN_DATA_SIZE* sizeof(float));
    if(err != cudaSuccess)
    {   
        cout<<"Could not allocate data for train labels \n";
        exit(-1);
    }
    err = cudaMemcpy(trainLabesDevice, training_dataset_labels, 10 * TRAIN_DATA_SIZE* sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {   
        cout<<"Could not train label  data to device \n";
        exit(-1);
    }


    file.open(testPath,  std::ifstream::ate | std::ifstream::binary);
    if(!file.is_open())
    {
        cout << "could not open mnist test csv \n";
        exit(-1);
    }

    size = file.tellg();
    file.seekg(ios_base::beg);
    file.read(dataset, size);
    file.close();
    loadDataset(training_dataset_buffer,training_dataset_labels,dataset, size);

    err = cudaMalloc(&testDataDevice, WIDTH * HEIGHT*TEST_DATA_SIZE * sizeof(float));
    if(err != cudaSuccess)
    {   
        cout<<"Could not allocate data for test images \n";
        exit(-1);
    }
    err = cudaMemcpy(testDataDevice, training_dataset_buffer,  WIDTH * HEIGHT*TEST_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {   
        cout<<"Could not copy test image data to device \n";
        exit(-1);
    }

    err = cudaMalloc(&testLabesDevice,10 * TEST_DATA_SIZE* sizeof(float));
    if(err != cudaSuccess)
    {   
        cout<<"Could not allocate data for test labels \n";
        exit(-1);
    }
    err = cudaMemcpy(testLabesDevice, training_dataset_labels, 10 * TEST_DATA_SIZE* sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {   
        cout<<"Could not load test label data to device \n";
        exit(-1);
    }

    file.close();
    delete[] dataset;
    delete[] training_dataset_labels;
    delete[] training_dataset_buffer;
}


int MnistDataset::parseNumber(char* dataset, int& startPos)
{
    int out_value = 0;
    char c = dataset[startPos];
    while (dataset[startPos] >= 48 && dataset[startPos] <= 57)
    {
        out_value*= 10;
        out_value += (dataset[startPos]- 48);
        startPos++;
    }
    
    return out_value;
}

void MnistDataset::print_image(float* values, float* labels)
{
    cout <<"[ ";
    for(int i=0; i < 10; i++)
         cout <<labels[i] << ", ";
    cout << "]\n";

    for(int y_w=0; y_w <28; y_w++)
    {
        for(int x_w = 0; x_w <28; x_w++)
        {
            if( values[y_w * 28 + x_w]== 0)
                cout <<"*";
            else 
                cout << "#";
        }
        cout << endl;
    }
    cout <<"----------------------------------------------\n" ;
    fflush(stdout);
}

void MnistDataset::buildBatches(unsigned int batch_size)
{
    batchSize = batch_size;
    trainBatchCount = TRAIN_DATA_SIZE/batchSize;
    testBatchCount = TEST_DATA_SIZE/batchSize;

    for(int i=0; i < TRAIN_DATA_SIZE; i++)
    {
        trainBatchAssignment.push_back(i);
        if(i < TEST_DATA_SIZE)
            testBatchAssignment.push_back(i);
    }
    trainBatchesImg = new Tensor*[trainBatchCount];
    trainBatchesLabes = new Tensor*[trainBatchCount];

    for(int i =0; i < trainBatchCount; i++)
    {
        trainBatchesImg[i] = new Tensor({batch_size, WIDTH * HEIGHT});
        trainBatchesLabes[i] = new Tensor({batch_size, 10});
    }

    testBatchesImg = new Tensor*[testBatchCount];
    testBatchesLabes = new Tensor*[testBatchCount];

    for(int i =0; i < testBatchCount; i++)
    {
        testBatchesImg[i] = new Tensor({batch_size, WIDTH * HEIGHT});
        testBatchesLabes[i] = new Tensor({batch_size, 10});
    }
    setTrainBatches();
    setTestBatches();
}

std::vector<Tensor*> MnistDataset::getTrainBatch(unsigned int i)
{
    return { trainBatchesImg[i], trainBatchesLabes[i]};
}

std::vector<Tensor*> MnistDataset::getTestBatch(unsigned int i)
{
    return { testBatchesImg[i], testBatchesLabes[i]};
}


void MnistDataset::shuffle()
{
    
}

void MnistDataset::loadDataset(float *images_dataset, float *labels_dataset, char *fileData, int maxSize)
{
    int i =0;
    //skip header
    while (fileData[i]!= '\n'){i++;}
    i++;

    int image_offset = 0;
    bool label = true;
    while (i < maxSize)
    {   
        if(fileData[i] == '\n' || fileData[i] == '\r' )
        {
            i++;
            continue;
        }

        int label = parseNumber(fileData, i);
        float arr[28*28];
        float labels[10] = {0,0,0,0,0,0,0,0,0,0};
        labels[label] = 1.0f;

        int number_count  = 0;
        while (number_count < 28*28)
        {
            i++;
            int number = parseNumber(fileData, i);
            float pixel = ((float)number)/255.0f;
            if(pixel > 0 )
            {
                int x  = 2;
            }
            arr[number_count] = pixel;
            number_count++;
        }
        memcpy(&images_dataset[image_offset *28*28], arr, sizeof(float) *28*28);
        memcpy(&labels_dataset[image_offset*10], labels, 10 * sizeof(float));
        image_offset++;
        i++;
    }
}

void MnistDataset::setTestBatches()
{
    for(int batch = 0; batch < testBatchCount;  batch++)
    {
        for(int sample = 0; sample < batchSize;  sample++)
        {
            unsigned int sampleIndex = testBatchAssignment[batch *batchSize + sample];

            testBatchesImg[batch]->setTensor_DeviceToDevice(testDataDevice + sampleIndex * WIDTH * HEIGHT * sizeof(float), 
                                                        WIDTH * HEIGHT* sizeof(float), sample * WIDTH * HEIGHT * sizeof(float));
            testBatchesLabes[batch]->setTensor_DeviceToDevice( testLabesDevice + sampleIndex * 10 * sizeof(float), 
                                                                                10* sizeof(float), sample * 10* sizeof(float));
        }
    }
}

void MnistDataset::setTrainBatches()
{
    for(int batch = 0; batch < trainBatchCount;  batch++)
    {
        for(int sample = 0; sample < batchSize;  sample++)
        {
            unsigned int sampleIndex = trainBatchAssignment[batch *batchSize + sample];

            trainBatchesImg[batch]->setTensor_DeviceToDevice(trainDataDevice + sampleIndex * WIDTH * HEIGHT * sizeof(float), 
                                                        WIDTH * HEIGHT* sizeof(float), sample * WIDTH * HEIGHT * sizeof(float));
            trainBatchesLabes[batch]->setTensor_DeviceToDevice( trainLabesDevice + sampleIndex * 10 * sizeof(float), 
                                                                                10* sizeof(float), sample * 10* sizeof(float));
        }
    }
}
