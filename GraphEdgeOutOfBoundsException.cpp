#include "GraphEdgeOutOfBoundsException.h"
#include <iostream>
#include <exception>
#include <string>
#include <sstream>

using namespace std;

GraphEdgeOutOfBoundsException::GraphEdgeOutOfBoundsException(int size, int edge) {
	cout << what(size, edge) << endl;
}

string GraphEdgeOutOfBoundsException::what(int size, int edge) const throw() {
	stringstream sstm;
	sstm << "Number of Edges (" << edge << ") exceded graph size (" << size << ")";
	string s = sstm.str();;
    return s;
}
