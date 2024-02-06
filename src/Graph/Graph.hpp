#include "../Expressions/Expression.hpp"
#include "../Expressions/Variable.hpp"
#include "../Expressions/Input.hpp"
#include <vector>

#ifndef GRAPH 
#define GRAPH

class Graph
{
public:
    Graph(Expression* graph);
    void compileGraph();
    void initVariables();
    void build();
    void execute();
private:
Expression* headOfGraph;
//executing order of executionList is 1st, 2nd, .... , n_th
std::vector<Expression*> executionList;
std::vector<Variable*> variableList;
std::vector<Input*> inputList;
};

#endif