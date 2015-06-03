#include "GraphArray.h"
#include "GraphVertexOutOfBoundsException.h"
#include "Path.h"
#include <iostream>
#include <algorithm>
#include "OriginalERGraph.h"
#include <cstdlib>
#include <ctime>

using namespace std;

int main () {
	/*
	GraphArray testGraph(10);
	cout << testGraph.getNumberOfVertices() << endl;

	testGraph.addEdge(2, 5);
	testGraph.addEdge(2, 8);
	testGraph.addEdge(3, 8);
	testGraph.printGraphAsArray();

	cout << testGraph.isDirectlyConnected(2, 4) << endl;
	cout << testGraph.isDirectlyConnected(8, 2) << endl;
	cout << testGraph.isConnected(5, 8) << endl;
	cout << testGraph.isDirectlyConnected(5, 8) << endl;
	cout << testGraph.isConnected(5, 3) << endl;

	OriginalERGraph er1(10, 0.8);
	er1.printGraphAsArray();

	GraphArray testGraph2(10);
	testGraph2.fillByBaselineER(50, 0.5);
	testGraph2.printGraphAsArray();
	cout << "number of edges = " << testGraph2.getNumberOfEdges() << endl;

	GraphArray testGraph3(10);
	testGraph3.fillByZER(100, 0.5);
	testGraph3.printGraphAsArray();
	cout << "number of edges = " << testGraph3.getNumberOfEdges() << endl;
*/
	GraphArray testGraph4(10);
	testGraph4.fillByPreZER(100, 0.5, 8);
	//testGraph4.clear();
	testGraph4.printGraphAsArray();
	cout << "number of edges = " << testGraph4.getNumberOfEdges() << endl;

    return 0;
}









