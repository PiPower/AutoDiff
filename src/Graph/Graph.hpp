#include "../Expressions/Expression.hpp"
#include "../Expressions/Variable.hpp"
#include "../Expressions/Input.hpp"
#include <vector>
#include <map>
#ifndef GRAPH 
#define GRAPH

class Graph
{
public:
    Graph(Expression* graph);
    void compileGraph();
    void build();
    void execute();
    void call(std::map<std::string, Tensor*>& inputs);
    void backwardPass();
private:
Expression* headOfGraph;
//executing order of executionList is 1st, 2nd, .... , n_th
std::vector<Expression*> executionList;
std::vector<Variable*> variableList;
std::vector<Input*> inputList;
};

#endif