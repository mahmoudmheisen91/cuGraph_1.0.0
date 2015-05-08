#include "GraphArray.h"
#include <iostream>
using namespace std;

int main () {
	GraphArray testGraph(10);
	//cout << testGraph.getNumberOfVertices() << endl;

	testGraph.addEdge(2, 5);
	testGraph.addEdge(2, 8);
	testGraph.printGraphAsArray();

	cout << testGraph.isConnected(2, 4) << endl;
	cout << testGraph.isConnected(8, 2) << endl;

    return 1;
}
