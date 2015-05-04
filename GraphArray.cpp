#include "GraphArray.hpp"
#include <iostream>
using namespace std;

GraphArray::GraphArray(int V) :numberOfVertices(V) {

}

GraphArray::~GraphArray() {
	//dtor
}

int GraphArray::getNumberOfVertices() {
	return numberOfvertices;
}

void GraphArray::setNumberOfVertices(int val) {
	numberOfVertices = val;
}

// when making const Object the function must be const
// by adding const keyword after the params list before { or ;
//void constObject::printing() const {

// member init list: is a must for initializing constant instanse variables
// :regVar(a), constVar(b) before { and nothing in the prototype (hpp)
// just define a and b as parameters both in implementation (cpp) and abstract (hpp)
// regVar and constVar are private variables
//constObject::constObject(int a, int b)
//   :constVar(b)

// default params must exits for all function params when used
// int volume(int l = 1, int w = 1, int h = 1);
