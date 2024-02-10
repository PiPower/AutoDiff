#include "../Expressions/Expression.hpp"
#include "../Expressions/Variable.hpp"
#include "../Expressions/Input.hpp"
#include <vector>
#include <map>
#ifndef GRAPH 
#define GRAPH

typedef std::map<std::string, Tensor*> FeedData;

class Graph
{
public:
    Graph(Expression* graph);
    void compileGraph();
    void build();
    void execute();
    void call(FeedData& inputs);
    Tensor* matchGradient(Expression* node, BackwardData& currentGradients);
    void backwardPass();
    void trainStep(FeedData& dataIn, float step, bool printLoss);
    void applyGradients(float* eta);
private:
Expression* headOfGraph;
BackwardData gradientRouteData;
//executing order of executionList is 1st, 2nd, .... , n_th
std::vector<Expression*> executionList;
std::vector<Variable*> variableList;
std::vector<Input*> inputList;
};

#endif