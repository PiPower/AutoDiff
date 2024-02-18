#include "Graph.hpp"
#include <stack>
#include <algorithm>
#include <iostream>
using namespace std;

Graph::Graph(Expression *graph, std::vector<Expression*> outputNodes, Optimizer* optimizer)
:
headOfGraph(graph), outputNodes(outputNodes),lastInferenceNode(-1), optimizer(optimizer)
{
}
/*
Compilation algorithm first searches for all input/variable node because they are always first
to exectute, but they should be executed only once during initialization period
Algorithm performs steps:
-During first graph pass add all input/variable expressions into variableList/inputList, all other types of nodes are added into waitList
-Loop until list is empty if all of the nodes are either variable/input type or marked as addedToExecutionList add node to 
exectution list, mark it as addedToExecutionList and remove it from waitList
*/
void Graph::compileGraph()
{
    stack<Expression*> nodes;
    vector<Expression*> waitList;
    nodes.push(headOfGraph);
    headOfGraph->visited = true;
    // first step
    while (nodes.size() > 0)
    {
        Expression* currentNode = nodes.top();
        nodes.pop();

        if( Variable *var = dynamic_cast<Variable*>(currentNode) )
        {
            variableList.push_back(var);
        }
        else if( Input *inp = dynamic_cast<Input*>(currentNode) )
        {
            inputList.push_back(inp);
        }
        else
        {
            waitList.push_back(currentNode);
        }

        for(Expression* child : currentNode->children)
        {
            if(!child->visited)
            {
                child->visited = true;
                nodes.push(child);
            }
        }
    }
    // second step
    int index = 0;
    while (waitList.size() > 0)
    {
        Expression* currentNode = waitList[index];
        bool readyForExecution = true;
        for(Expression* child : currentNode->children)
        {
            if(!(dynamic_cast<Variable*>(child) || dynamic_cast<Input*>(child) || child->addedToExecutionList) )
            {
                readyForExecution= false;
                break;
            }
        }

        if(readyForExecution)
        {
            currentNode->addedToExecutionList = true;
            executionList.push_back(currentNode);
            vector<Expression*>::iterator currentNodeIter = waitList.begin() + index;
            waitList.erase(currentNodeIter);
        }

        index++;
        if(index >= waitList.size()) 
        {
            index = 0;
        }
    }

    // check if all output nodes belong to execution list
    for(Expression* outputNode : outputNodes)
    {
       auto iter = find(executionList.begin(), executionList.end(), outputNode);
       if(iter == executionList.end())
       {
            logErrorAndExit(true, "node in output nodes does not belong to the Graph");
       }
    }


    // find the last node to execture in inference call
    for(int i = executionList.size() - 1; i >=0; i-- )
    {
       auto iter = find(outputNodes.begin(), outputNodes.end(), executionList[i]);
       if(iter != outputNodes.end())
       {
            lastInferenceNode = i;
       }
    }
    
}

/*
inits all variable nodes. 
*/


/*
Creates all the required context for cuda and possibly other libraries. 
Calling more than once will cause memory leaks and may cause undefined behaviour
*/
void Graph::build()
{
    for(Variable* node : variableList)
    {
        node->build();
    }
    for(Expression* node : executionList)
    {
        node->build();
    }

    optimizer->build(variableList);

    cudaDeviceSynchronize();
    Tensor::streamSync();
}

void Graph::trainCall(std::map<std::string, Tensor*>& inputs)
{
    for(Input* input : inputList)
    {
        const string* name = input->getName();
        auto iterator = inputs.find(*name);
        if(iterator == inputs.cend())
        {
            cerr<<"Given input does not contrain tensor for input node: " << *name << endl;
            exit(-1);
        }
        Tensor* in = iterator->second;
        input->setInput(in);
    }
    
    for(Expression* node : executionList)
    {
        node->execute();
    }
}

Tensor *Graph::matchGradient(Expression *node, BackwardData &currentGradients)
{
    Tensor* gradOut = nullptr;
    auto& grads = currentGradients.gradientTensors;
    auto& nodes = currentGradients.nodeAddres;

    int i = 0;
    for(; i < grads.size(); i++)
    {
        if( nodes[i] == node)
        {
            gradOut = currentGradients.gradientTensors[i];
            break;
        }
    }

    for(; i < grads.size(); i++)
    {
        if( nodes[i] == node)
        {
            Tensor::addTensors(gradOut,gradOut, currentGradients.gradientTensors[i]);
        }
    }

    return gradOut;
}

void Graph::backwardPass()
{
    Tensor* grad = Tensor::createWithConstant(1.0f, {});

    for(int i=executionList.size()-1; i >=0; i--)
    {   
        executionList[i]->backwardPass(grad, gradientRouteData);
        if(i > 0)
        {
            grad = matchGradient(executionList[i-1], gradientRouteData);
            logErrorAndExit(grad == nullptr, "No gradient for op node\n");
        }
    }
    //all the remaining gradient belong to Variables/Inputs
}

void Graph::trainStep(FeedData &dataIn, bool printLoss)
{
    trainCall(dataIn);
    if(printLoss) executionList[executionList.size()-1]->getTensor()->printTensor(stdout);
    backwardPass();
    applyGradients();

    //sync execution and clear all grads 
    Tensor::streamSync();
    for(int i=0 ; i < gradientRouteData.gradientTensors.size(); i ++)
    {
        delete  gradientRouteData.gradientTensors[i];
    }

    gradientRouteData.gradientTensors.clear();
    gradientRouteData.nodeAddres.clear();
}

void Graph::applyGradients()
{
    optimizer->nextLoop();
    for(Variable* var : variableList)
    {
        Tensor* grad = matchGradient(var, gradientRouteData);
        logErrorAndExit(grad == nullptr, "No gradient for variable node\n");
        optimizer->updateGradient(var, grad);
        var->applyGradients(grad);
    }
}

std::vector<Tensor *> Graph::inferenceCall(FeedData& dataIn)
{
    std::vector<Tensor*> output;

    for(Input* input : inputList)
    {   

        if(input->isLabel())
        {
            continue;
        }
        const string* name = input->getName();
        auto iterator = dataIn.find(*name);
        if(iterator == dataIn.cend())
        {
            cerr<<"Given input does not contrain tensor for input node: " << *name << endl;
            exit(-1);
        }
        Tensor* in = iterator->second;
        input->setInput(in);
    }

    for(int i =0; i <= lastInferenceNode; i++ )
    {
        executionList[i]->execute();
    }

    Tensor::streamSync();
    for(Expression* outputNode : outputNodes)
    {
        Tensor *t = new Tensor(*outputNode->getTensor()) ;
        output.push_back(t);
    }

    return output;
}
