#include "IncludeExpressions.hpp"
#include "../../src/Graph/Graph.hpp"
#include "../../src/Utils/Initializers.hpp"
#include "../../src/Tensor/Tensor.hpp"
#include "./MnistDataset.hpp"
#include <iostream>
#include <string>

using namespace std;
int main()
{

    MnistDataset dataset("/mnt/e/mnist_train.csv", "/mnt/e/mnist_test.csv");
    dataset.buildBatches(64);


    string input_name = "entry";
    string label_name = "labels";
    FeedData data;
    GaussianInitializer init(0,1, TensorType::float32);
    //ConstantInitializer init(1, TensorType::float32);

    Expression* model;

    Input* v1 = new Input({64,784}, input_name);
    Variable* v2 = new Variable({784, 128}, &init);
    model = new Matmul(v1, v2);
    model = new Activation(model, ActivationType::sigmoid);

    Variable* v3 = new Variable({128, 128}, &init);
    model = new Matmul(model, v3);
    model = new Activation(model, ActivationType::sigmoid);

    Variable* v4 = new Variable({128, 10}, &init);
    model = new Matmul(model, v4);
    Softmax* model_soft = new Softmax(model, {1});

    Input* labels = new Input({64,10}, "labels");
    CategoricalCrossentropy* loss = new CategoricalCrossentropy(model_soft, labels, {1});
    ReduceMean* sum = new ReduceMean(loss, {0,1});

    Graph g(sum);

    g.compileGraph();
    g.build();
    g.call(data);
    
    for(int i=0; i < 937; i++)
    {
        data[label_name] = dataset.getBatch(i)[1];
        data[input_name] = dataset.getBatch(i)[0];
        
        g.trainStep(data, 0.001, (i + 1)% 50 == 0);
        if((i+1)% 50 == 0 )
        {
            cout << "it is batch number :" << i << endl;
            fflush(stdout);
        }
    }
    

    //sum->getTensor()->printTensor(stdout);
}