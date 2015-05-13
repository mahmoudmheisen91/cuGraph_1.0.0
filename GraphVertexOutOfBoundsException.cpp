#include "GraphVertexOutOfBoundsException.h"
#include "GraphArray.h"
#include <iostream>
#include <exception>
#include <string>

using namespace std;

GraphVertexOutOfBoundsException::GraphVertexOutOfBoundsException(GraphArray g, int vert) {
	cout << what(g, vert) << endl;
}

const char* GraphVertexOutOfBoundsException::what(GraphArray g, int vert) const throw() {
	string s = "Vertix " + vert + "is outside of graph range [0 - " + g.getNumberOfVertices() + "]";
    return s;
}
