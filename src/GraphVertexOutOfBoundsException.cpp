#include "GraphVertexOutOfBoundsException.h"

GraphVertexOutOfBoundsException::GraphVertexOutOfBoundsException() {
	//ctor
}

GraphVertexOutOfBoundsException::~GraphVertexOutOfBoundsException() {
	//dtor
}

virtual const char* what() const throw() {
    return "My exception happened";
}
