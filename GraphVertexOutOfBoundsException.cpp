#include "GraphVertexOutOfBoundsException.h"
#include <iostream>
#include <exception>
using namespace std;

GraphVertexOutOfBoundsException::GraphVertexOutOfBoundsException() {
	//cout << what() << endl;
}

const char* GraphVertexOutOfBoundsException::what() const throw() {
    return "My exception happened";
}
