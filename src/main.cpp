#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#include "draw.h"
#include "Graph.h"

using namespace std;

int main () {
    Graph g1(10);
    //g1.setType(DIRECTED, SELF_LOOP);
    g1.fillByBaselineER(100, 0.5);
    //cout << g1.getNumberOfVertices() << endl;
    //cout << g1.getNumberOfEdges() << endl;

    //g1.addEdge(2, 5);
    //g1.addEdge(2, 8);
    //g1.addEdge(3, 8);
    //g1.printGraphAsArray();

    //cout << g1.isDirectlyConnected(2, 5) << endl;
    //cout << g1.isDirectlyConnected(8, 2) << endl;
    //cout << g1.isConnected(5, 8) << endl;
    //cout << g1.isDirectlyConnected(5, 8) << endl;
    //cout << g1.isConnected(5, 3) << endl;
    //cout << g1.getNumberOfEdges() << endl;
    g1.draw();

	/*
	GraphArray testGraph2(10);
	testGraph2.fillByBaselineER(50, 0.5);
	testGraph2.printGraphAsArray();
	cout << "number of edges = " << testGraph2.getNumberOfEdges() << endl;

	GraphArray testGraph3(10);
	testGraph3.fillByZER(100, 0.5);
	testGraph3.printGraphAsArray();
	cout << "number of edges = " << testGraph3.getNumberOfEdges() << endl;

    Graph testGraph4(10);
	testGraph4.fillByPreZER(100, 0.5, 8);
    testGraph4.clear();
	testGraph4.printGraphAsArray();
	cout << "number of edges = " << testGraph4.getNumberOfEdges() << endl;
    */
    return 0;
}









