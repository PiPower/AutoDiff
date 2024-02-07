#include "Graph.hpp"
#include <stack>
#include <algorithm>
#include <iostream>
using namespace std;

Graph::Graph(Expression *graph)
:
headOfGraph(graph)
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
            if(!(dynamic_cast<Variable*>(child) || child->addedToExecutionList) )
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
}

void Graph::execute()
{
    for(Expression* node : executionList)
    {
        node->execute();
    }
}

void Graph::call(std::map<std::string, Tensor*>& inputs)
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
    
    execute();
}
