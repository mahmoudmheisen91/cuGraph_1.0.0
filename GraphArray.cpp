#include "GraphArray.h"
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
	content[v1 * numberOfVertices + v2] = 1;
	content[v2 * numberOfVertices + v1] = 1;
}

bool GraphArray::isConnected(int v1, int v2) {
	if (content[v1 * numberOfVertices + v2] && content[v2 * numberOfVertices + v1])
		return true;
	return false;
}

void GraphArray::checkVertixName(int vert) {

}






