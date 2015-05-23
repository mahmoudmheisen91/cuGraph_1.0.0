#include "OriginalERGraph.h"
#include "GraphArray.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

OriginalERGraph::OriginalERGraph(int V, double p) {
	graph = new GraphArray(V);

	srand(time(0));
	double theta;
	for(int i = 0; i < V; i++) {
		for(int j = 0; j < V && j != i; j++) {
			theta = (double)rand() / RAND_MAX;

			if (theta > p)
				graph->addEdge(i, j);
		}
	}

}

OriginalERGraph::~OriginalERGraph() {
	delete graph;
}

int OriginalERGraph::getNumberOfVertices() {
	return graph->getNumberOfVertices();
}

int OriginalERGraph::getNumberOfEdges() {
	return graph->getNumberOfEdges();
}

int OriginalERGraph::getSize() {
	return graph->getSize();
}

void OriginalERGraph::printGraphAsArray(void) {
	graph->printGraphAsArray();
}

void OriginalERGraph::addEdge(int v1, int v2) {
	graph->addEdge(v1, v2);
}

void OriginalERGraph::removeEdge(int v1, int v2) {
	graph->removeEdge(v1, v2);
}

bool OriginalERGraph::isDirectlyConnected(int v1, int v2) {
	return graph->isDirectlyConnected(v1 , v2);
}

bool OriginalERGraph::isConnected(int v1, int v2) {
	return graph->isConnected(v1, v2);
}





