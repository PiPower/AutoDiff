#include "IncludeExpressions.hpp"
#include "../../src/Graph/Graph.hpp"
#include "../../src/Graph/Optimizers/SGD.hpp"
#include "../../src/Utils/Initializers.hpp"
#include "../../src/Tensor/Tensor.hpp"
#include "./MnistDataset.hpp"
#include <iostream>
#include <string>

using namespace std;

int compare(float* predicted, float* real);

int main()
{
    MnistDataset dataset("/mnt/e/mnist_train.csv", "/mnt/e/mnist_test.csv");
    dataset.buildBatches(64);

    string input_name = "entry";
    string label_name = "labels";
    FeedData data;

    Expression* model;
    
    Input* v1 = new Input({64,784}, input_name);
    model = new Reshape(v1, {64, 1, 28, 28});

    GlorotUniform v2_init(32, 1, TensorType::float32);
    Variable* v2 = new Variable({32, 1, 3, 3}, &v2_init);
    model = new Conv2D(model, v2, {1,1}, {1,1} );
    model = new Pooling2D(model, PoolingType::max);


    GlorotUniform v5_init(32, 64, TensorType::float32);
    Variable* v5 = new Variable({64, 32, 3, 3}, &v2_init);
    model = new Conv2D(model, v5, {1,1}, {1,1} );
    model = new Pooling2D(model, PoolingType::max);

    model = new Reshape(model, {64, 7*7*64});

    GlorotUniform v3_init(7*7*64, 32, TensorType::float32);
    Variable* v3 = new Variable({7*7*64, 32}, &v3_init);
    model = new Matmul(model, v3);
    model = new Activation(model, ActivationType::sigmoid);

    GlorotUniform v4_init(32, 10, TensorType::float32);
    Variable* v4 = new Variable({32, 10}, &v4_init);
    model = new Matmul(model, v4);
    Softmax* model_soft = new Softmax(model, {1});

    Input* labels = new Input({64,10}, "labels", true);
    CategoricalCrossentropy* loss = new CategoricalCrossentropy(model_soft, labels, {1});
    ReduceMean* sum = new ReduceMean(loss, {0,1});

    SGD* optimizer = new SGD(0.01);
    Graph g(sum, {model_soft}, optimizer);

    g.compileGraph();
    g.build();
   
    int epochs = 5;
    for(int e =0; e < epochs; e++)
    {
        for(int i=0; i < dataset.getTrainBatchCount(); i++)
        {
            data[input_name] = dataset.getTrainBatch(i)[0];
            data[label_name] = dataset.getTrainBatch(i)[1];

            g.trainStep(data, (i+1)% 50 == 0);
            if((i+1)% 50 == 0 )
            {
                cout << "it is batch number :" << i + 1 << " epoch number: "<< e + 1 << endl;
                fflush(stdout);
            }
        }
       dataset.shuffle();
    }
    
    FeedData dataInference;
    float accuracy = 0;
    for(int i=0; i < dataset.getTestBatchCount() ; i++)
    {
            dataInference[input_name] = dataset.getTestBatch(i)[0];

            std::vector<Tensor*> out = g.inferenceCall(dataInference);
            float* predicted = (float*) out[0]->getTensorValues();
            float* real = (float*)dataset.getTestBatch(i)[1]->getTensorValues();

            for(int i = 0; i < dataset.getBatchSize(); i++)
            {
                accuracy += compare( predicted + i*10, real + i*10 );
            } 
            delete[] predicted;
            delete[] real;
    }
    cout << "Accuracy: " << accuracy/10000.0f << endl;
}

int compare(float *predicted, float *real)
{
    int argmax = 0;
    for(int i = 1; i < 10; i++)
    {
        if(predicted[i] > predicted[argmax] ) argmax =i;
    }
    
    return  real[argmax] == 1? 1 : 0;
}
