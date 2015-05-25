#include "GraphArray.h"
#include "GraphVertexOutOfBoundsException.h"
#include "Path.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

GraphArray::GraphArray() {}

GraphArray::GraphArray(int V) :numberOfVertices(V) {
	size = numberOfVertices * numberOfVertices;
	numberOfEdges = 0;
	content = new int[size];

	fill(content, content+size, 0);
}

GraphArray::~GraphArray() {
	delete content;
}

int GraphArray::getNumberOfVertices() {
	return numberOfVertices;
}

int GraphArray::getNumberOfEdges() {
	return numberOfEdges;
}

int *GraphArray::getContent() {
	return content;
}

int GraphArray::getSize() {
	return size;
}

void GraphArray::printGraphAsArray(void) {
    for(int i = 1; i <= size; i++) {
    	cout << content[i-1] << " ";

        if (i % numberOfVertices == 0)
			cout << endl;
    }
}

void GraphArray::addEdge(int v1, int v2) {
	checkVertixName(v1);
	checkVertixName(v2);
	content[v1 * numberOfVertices + v2] = 1;
	content[v2 * numberOfVertices + v1] = 1;
	numberOfEdges++;
}

void GraphArray::removeEdge(int v1, int v2) {
	checkVertixName(v1);
	checkVertixName(v2);
	content[v1 * numberOfVertices + v2] = 0;
	content[v2 * numberOfVertices + v1] = 0;
	numberOfEdges--;
}

bool GraphArray::isDirectlyConnected(int v1, int v2) {
	checkVertixName(v1);
	checkVertixName(v2);
	if (content[v1 * numberOfVertices + v2] && content[v2 * numberOfVertices + v1])
		return true;
	return false;
}

bool GraphArray::isConnected(int v1, int v2) {
	checkVertixName(v1);
	checkVertixName(v2);

	Path p(this, v1);
	return p.hasPathTo(v2);
}

void GraphArray::fillByBaselineER(int E, double p) {
	srand(time(0));
	double theta;

	int l, m;
	for(int i = 0; i < E; i++) {
		theta = (double)rand() / RAND_MAX;

		if (theta < p) {
			l = i / numberOfVertices;
			m = i % numberOfVertices;
			addEdge(l, m);
		}
	}
}

void GraphArray::fillByZER(int E, double p) {
	srand(time(0));
	double theta, logp;

	int l, m, k, i = -1;
	while (i < E) {
		theta = (double)rand() / RAND_MAX;
		logp = log10f(theta)/log10f(1-p);

		k = max(0, (int)ceil(logp) - 1);
		i += k + 1;

		if(i < E) { // equavelent to: Discard last edge, because i > E
			l = i / numberOfVertices;
			m = i % numberOfVertices;
			addEdge(l, m);
		}
	}
}

void GraphArray::checkVertixName(int vert) {
	if (vert < 0 || vert >= numberOfVertices)
		throw new GraphVertexOutOfBoundsException(numberOfVertices, vert);
}





