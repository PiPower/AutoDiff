#include "IncludeExpressions.hpp"
#include "../../src/Graph/Graph.hpp"
#include "../../src/Graph/Optimizers/Adam.hpp"
#include "../../src/Utils/Initializers.hpp"
#include "../../src/Tensor/Tensor.hpp"
#include "./MnistDataset.hpp"
#include <iostream>
#include <string>

using namespace std;

int compare(float* predicted, float* real);
Expression* createResNet(unsigned int channels, unsigned int width, 
                unsigned int height, unsigned int startDepth, unsigned int batchSize, unsigned int layers);

int main()
{
    MnistDataset dataset("/mnt/e/mnist_train.csv", "/mnt/e/mnist_test.csv");
    dataset.buildBatches(64);

    FeedData data;
    
    Expression* Resnet = createResNet(1,28,28,64,64,1);

    Input* labels = new Input({64,10}, "labels", true);
    CategoricalCrossentropy* loss = new CategoricalCrossentropy(Resnet, labels, {1});
    ReduceMean* sum = new ReduceMean(loss, {0,1});

    Adam* optimizer = new Adam(0.001);
    Graph g(sum, {Resnet}, optimizer);

    g.compileGraph();
    g.build();
   
    int epochs = 2;
    for(int e =0; e < epochs; e++)
    {
        for(int i=0; i < dataset.getTrainBatchCount(); i++)
        {
            data["entry"] = dataset.getTrainBatch(i)[0];
            data["labels"] = dataset.getTrainBatch(i)[1];

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
            dataInference["entry"] = dataset.getTestBatch(i)[0];

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

Expression *createResNet(unsigned int channels, unsigned int width, 
                unsigned int height, unsigned int startDepth, unsigned int batchSize, unsigned int layers)
{
    string input_name = "entry";
    string label_name = "labels";
    Expression* model;

    Input* v1 = new Input({batchSize,channels * width * height}, input_name);
    model = new Reshape(v1, {batchSize, channels, height, width});

    GlorotUniform* v2_init = new GlorotUniform(startDepth/2, channels, TensorType::float32);
    Variable* v2 = new Variable({startDepth/2 , channels, 7, 7}, v2_init);
    model = new Conv2D(model, v2, {1,1}, {3,3} );
    model = new Activation(model, ActivationType::relu);
    model = new Pooling2D(model, PoolingType::max);

    unsigned int oldDepth = startDepth/2;
    for(int i = 0; i < layers; i++)
    {   
        unsigned int newDepth = startDepth * (unsigned int)pow(2, i);

        GlorotUniform* weight_init = new GlorotUniform(newDepth, oldDepth, TensorType::float32);
        Variable* weight = new Variable({newDepth, oldDepth, 3, 3}, weight_init);
        model = new Conv2D(model, weight, {1,1}, {1,1} );
        Expression* model_skip_connection = new Activation(model, ActivationType::relu);


        GlorotUniform* weight_init2 = new GlorotUniform(newDepth, newDepth, TensorType::float32);
        weight = new Variable({newDepth, newDepth, 3, 3}, weight_init2);
        model = new Conv2D(model_skip_connection, weight, {1,1}, {1,1} );
        model = new Activation(model, ActivationType::relu);


        model = new Addition(model, model_skip_connection);


        weight = new Variable({newDepth, newDepth, 3, 3}, weight_init2);
        model = new Conv2D(model, weight, {1,1}, {1,1} );
        model_skip_connection = new Activation(model, ActivationType::relu);


        weight = new Variable({newDepth, newDepth, 3, 3}, weight_init2);
        model = new Conv2D(model, weight, {1,1}, {1,1} );
        model = new Activation(model, ActivationType::relu);


        model = new Addition(model, model_skip_connection);


        weight = new Variable({newDepth, newDepth, 3, 3}, weight_init2);
        model = new Conv2D(model, weight, {1,1}, {1,1} );
        model_skip_connection = new Activation(model, ActivationType::relu);


        weight = new Variable({newDepth, newDepth, 3, 3}, weight_init2);
        model = new Conv2D(model, weight, {1,1}, {1,1} );
        model = new Activation(model, ActivationType::relu);

        model = new Addition(model, model_skip_connection);

        weight = new Variable({newDepth, newDepth, 3, 3}, weight_init2);
        model = new Conv2D(model, weight, {2,2}, {1,1} );
        model = new Activation(model, ActivationType::sigmoid);

        oldDepth = newDepth;
    }
    unsigned int dims = oldDepth * (width / ((unsigned int)pow(2, layers + 1)) ) * (height / ((unsigned int)pow(2, layers + 1)) );

    GlorotUniform* denseInit = new GlorotUniform(10, dims, TensorType::float32);
    Variable* w = new Variable({dims, 10}, denseInit);
    model = new Reshape(model, {batchSize, dims});
    model = new Matmul(model, w);
    model = new Softmax(model, {1});

    return model;
}
