#ifndef GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
#define GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_

#include <exception>
#include <string>
using namespace std;

class GraphVertexOutOfBoundsException: public exception
{
	public:
		GraphVertexOutOfBoundsException(int size, int vert);
		virtual string what(int size, int vert) const throw();
	protected:
	private:
};

#endif // GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
