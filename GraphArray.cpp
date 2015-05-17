#include "GraphArray.h"
#include "GraphVertexOutOfBoundsException.h"
//#include "Path.h"
#include <iostream>
#include <algorithm>

using namespace std;

GraphArray::GraphArray(int V) :numberOfVertices(V) {
	size = numberOfVertices * numberOfVertices;
	content = new int[size];

	fill(content, content+size, 0);
}

GraphArray::~GraphArray() {
	delete content;
}

//GraphArray::GraphArray(const GraphArray& x)
////	: numberOfVertices(x.numberOfVertices), size(x.size), content(x.content)
//{}

int GraphArray::getNumberOfVertices() {
	return numberOfVertices;
}

void GraphArray::printGraphAsArray(void) {
	int row = numberOfVertices;
	int gSize = numberOfVertices * numberOfVertices;

    for(int i = 1; i <= gSize; i++) {
    	cout << content[i-1] << " ";

        if (i % row == 0)
			cout << endl;
    }
}

void GraphArray::addEdge(int v1, int v2) {
	checkVertixName(v1);
	checkVertixName(v2);
	content[v1 * numberOfVertices + v2] = 1;
	content[v2 * numberOfVertices + v1] = 1;
}

bool GraphArray::isDirectlyConnected(int v1, int v2) {
	checkVertixName(v1);
	checkVertixName(v2);
	if (content[v1 * numberOfVertices + v2] && content[v2 * numberOfVertices + v1])
		return true;
	return false;
}

//bool GraphArray::isConnected(int v1, int v2) {
//	checkVertixName(v1);
//	checkVertixName(v2);

//	Path p(*this, v1);
//	return p.hasPathTo(v2);
//}

void GraphArray::checkVertixName(int vert) {
	if (vert < 0 || vert >= numberOfVertices)
		throw new GraphVertexOutOfBoundsException(numberOfVertices, vert);
}





