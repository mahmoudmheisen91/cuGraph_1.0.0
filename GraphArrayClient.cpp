#include "GraphArray.h"
#include "GraphVertexOutOfBoundsException.h"
#include "Path.h"
#include <iostream>
#include <algorithm>

bool isConnected(GraphArray *g, int v1, int v2);

using namespace std;

int main () {
	GraphArray testGraph(10);
	cout << testGraph.getNumberOfVertices() << endl;

	testGraph.addEdge(2, 5);
	testGraph.addEdge(2, 8);
	testGraph.printGraphAsArray();

	cout << testGraph.isDirectlyConnected(2, 4) << endl;
	cout << testGraph.isDirectlyConnected(8, 2) << endl;
	cout << isConnected(&testGraph, 5, 8) << endl;

    return 0;
}

bool isConnected(GraphArray *g, int v1, int v2) {

	Path p(g, v1);
	return p.hasPathTo(v2);
}

