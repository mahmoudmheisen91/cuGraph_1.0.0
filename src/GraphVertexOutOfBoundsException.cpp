#include "GraphVertexOutOfBoundsException.h"
#include <iostream>
#include <exception>
#include <string>
#include <sstream>

using namespace std;

GraphVertexOutOfBoundsException::GraphVertexOutOfBoundsException(int size, int vert) {
	cout << what(size, vert) << endl;
}

string GraphVertexOutOfBoundsException::what(int size, int vert) const throw() {
	stringstream sstm;
	sstm << "Vertix " << vert << " is outside of graph range [0 >> " << size-1 << "]";
	string s = sstm.str();;
    return s;
}
