#include "GraphArray.h"
#include "GraphVertexOutOfBoundsException.h"
#include "GraphEdgeOutOfBoundsException.h"
#include "Path.h"
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

GraphArray::GraphArray() {}

GraphArray::GraphArray(int V) :numberOfVertices(V) {
	size = pow(numberOfVertices, 2);
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
	if(isFullyConnected()) // method
//		throw new GraphEdgeOutOfBoundsException(size, edge); //change with other exception

	checkVertixName(v1);
	checkVertixName(v2);
	content[v1 * numberOfVertices + v2] = 1;
	content[v2 * numberOfVertices + v1] = 1;
	numberOfEdges++;
}

void GraphArray::removeEdge(int v1, int v2) {
	if(isEmpty()) // method
//		throw new GraphEdgeOutOfBoundsException(size, edge); //change with other exception

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

bool GraphArray::isFullyConnected() {
	if(numberOfEdges == pow(numberOfVertices, 2))
		return true;

	return false;
}

bool GraphArray::isEmpty() { // from edge prespective not vertix (because vertices is constant (cuda))
	if(numberOfEdges == 0)
		return true;

	return false;
}

void GraphArray::fillByBaselineER(int E, double p) {
	checkEdgeRange(E);
	srand(time(0));
	double theta;

	int v1, v2;
	for(int i = 0; i < E; i++) {
		theta = (double)rand() / RAND_MAX;

		if (theta < p) {
			v1 = i / numberOfVertices;
			v2 = i % numberOfVertices;
			addEdge(v1, v2);
		}
	}
}

void GraphArray::fillByZER(int E, double p) {
	checkEdgeRange(E);
	srand(time(0));
	double theta, logp;

	int v1, v2, k, i = -1;
	while (i < E) {
		theta = (double)rand() / RAND_MAX;
		logp = log10f(theta)/log10f(1-p);

		k = max(0, (int)ceil(logp) - 1);
		i += k + 1;

		if(i < E) { // equavelent to: Discard last edge, because i > E
			v1 = i / numberOfVertices;
			v2 = i % numberOfVertices;
			addEdge(v1, v2);
		}
	}
}

void GraphArray::fillByPreZER(int E, double p, int m) {
	checkEdgeRange(E);
	srand(time(0));
	double theta, logp;
	double *F = new double[m+1];

	for(int i = 0; i <= m; i++) {
		F[i] = 1 - pow(1-p, i+1);
	}

	int v1, v2, k, j, i = -1;
	while(i < E) {
		theta = (double)rand() / RAND_MAX;

		j = 0;
		while(j <= m) {
			if(F[j] > theta) {
				k = j;
				break; // must break from j while loop not i while loop
			}
			j++;
		}

		// if could not find k from the upper loop
		if(j == m+1) { // rare to happen for large m value
			logp = log10f(1-theta)/log10f(1-p);
			k = max(0, (int)ceil(logp) - 1);
		}

		i += k + 1;

		if(i < E) { // equavelent to: Discard last edge, because i > E
			v1 = i / numberOfVertices;
			v2 = i % numberOfVertices;
			addEdge(v1, v2);
		}
	}
}

void GraphArray::checkVertixName(int vert) {
	if (vert < 0 || vert >= numberOfVertices)
		throw new GraphVertexOutOfBoundsException(numberOfVertices, vert);
}

void GraphArray::checkEdgeRange(int edge) {
	if (edge < 0 || edge > size)
		throw new GraphEdgeOutOfBoundsException(size, edge);
}




