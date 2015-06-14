#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include "Exceptions.h"

using namespace std;

namespace cuGraph {
    GraphVertexOutOfBoundsException::GraphVertexOutOfBoundsException(int size, int vert) {
        cout << what(size, vert) << endl;
    }

    string GraphVertexOutOfBoundsException::what(int size, int vert) const throw() {
        stringstream sstm;
        sstm << "Vertix " << vert << " is outside of graph range [0 >> " << size-1 << "]";
        string s = sstm.str();
        return s;
    }

    GraphEdgeOutOfBoundsException::GraphEdgeOutOfBoundsException(int size, int edge) {
        cout << what(size, edge) << endl;
    }

    string GraphEdgeOutOfBoundsException::what(int size, int edge) const throw() {
        stringstream sstm;
        sstm << "Number of Edges (" << edge << ") exceded graph size (" << size << ")";
        string s = sstm.str();
        return s;
    }

    GraphNumberOfVertexOutOfBoundsException::GraphNumberOfVertexOutOfBoundsException(int verts) {
        cout << what(verts) << endl;
    }

    string GraphNumberOfVertexOutOfBoundsException::what(int verts) const throw() {
        stringstream sstm;
        sstm << "Number of Vertixes (" << verts << ") is <= 0 (";
        string s = sstm.str();
        return s;
    }

    GraphIsFullException::GraphIsFullException(void) {
        cout << what(0) << endl;
    }

    string GraphIsFullException::what(int) const throw() {
        stringstream sstm;
        sstm << "Graph is Fully connected!!";
        string s = sstm.str();
        return s;
    }

    GraphIsEmptyException::GraphIsEmptyException(void) {
        cout << what(0) << endl;
    }

    string GraphIsEmptyException::what(int) const throw() {
        stringstream sstm;
        sstm << "Graph is Empty!!";
        string s = sstm.str();
        return s;
    }
}
