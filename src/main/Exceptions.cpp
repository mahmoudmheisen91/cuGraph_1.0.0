/*
 * Exceptions.cpp
 *
 *  Created: 2015-05-24, Modified: 2015-08-11
 *
 */
 
// Standard C++ libraries includes:
#include <iostream>
#include <sstream>

// Headers includes:
#include <main/Exceptions.h>

namespace cuGraph {
	// Throw this exception, if the vertix out of bound (0 - max-1):
    GraphVertexOutOfBoundsException::GraphVertexOutOfBoundsException(long long int size, int vert) {
        cout << what(size, vert) << endl;
    }

    string GraphVertexOutOfBoundsException::what(long long int size, int vert) const throw() {
        stringstream sstm;
        sstm << "Vertix " << vert << " is outside of graph range [0 >> " << size-1 << "]";
        string s = sstm.str();
        return s;
    }
	
	// Throw this exception, if the edge count exceeded max size:
    GraphEdgeOutOfBoundsException::GraphEdgeOutOfBoundsException(long long int size, int edge) {
        cout << what(size, edge) << endl;
    }

    string GraphEdgeOutOfBoundsException::what(long long int size, int edge) const throw() {
        stringstream sstm;
        sstm << "Number of Edges (" << edge << ") exceded graph size (" << size << ")";
        string s = sstm.str();
        return s;
    }

	// Throw this exception, if the vertix size exceeded max size:
    GraphNumberOfVertexOutOfBoundsException::GraphNumberOfVertexOutOfBoundsException(long long int verts) {
        cout << what(verts) << endl;
    }

    string GraphNumberOfVertexOutOfBoundsException::what(long long verts) const throw() {
        stringstream sstm;
        sstm << "Number of Vertixes (" << verts << ") is <= 0 (";
        string s = sstm.str();
        return s;
    }

	// Throw this exception, if the graph is full:
    GraphIsFullException::GraphIsFullException(void) {
        cout << what(0) << endl;
    }

    string GraphIsFullException::what(int) const throw() {
        stringstream sstm;
        sstm << "Graph is Fully connected!!";
        string s = sstm.str();
        return s;
    }

	// Throw this exception, if the graph is empty:
    GraphIsEmptyException::GraphIsEmptyException(void) {
        cout << what(0) << endl;
    }

    string GraphIsEmptyException::what(int) const throw() {
        stringstream sstm;
        sstm << "Graph is Empty!!";
        string s = sstm.str();
        return s;
    }

	// Throw this exception, if the direction is undefined:
    GraphDirectionTypeException::GraphDirectionTypeException(void) {
        cout << what(0) << endl;
    }

    string GraphDirectionTypeException::what(int) const throw() {
        stringstream sstm;
        sstm << "Graph Direction Type is wrong!!";
        string s = sstm.str();
        return s;
    }
	
	// Throw this exception, if the graph is not initalized:
    GraphIsNotInitException::GraphIsNotInitException(void) {
        cout << what(0) << endl;
    }

    string GraphIsNotInitException::what(int) const throw() {
        stringstream sstm;
        sstm << "Graph is not initialized!!";
        string s = sstm.str();
        return s;
    }
}
