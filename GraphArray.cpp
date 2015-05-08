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

void GraphArray::setNumberOfVertices(int val) {
	numberOfVertices = val;
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

// when making const Object the function must be const
// by adding const keyword after the params list before { or ;
//void constObject::printing() const {

// member init list: is a must for initializing constant instanse variables
// :regVar(a), constVar(b) before { and nothing in the prototype (hpp)
// just define a and b as parameters both in implementation (cpp) and abstract (hpp)
// regVar and constVar are private variables
//constObject::constObject(int a, int b)
//   :constVar(b)

// default params must exits for all function params when used
// int volume(int l = 1, int w = 1, int h = 1);
