#ifndef GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
#define GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_

#include <exception>
using namespace std;

class GraphVertexOutOfBoundsException: public exception
{
	public:
		GraphVertexOutOfBoundsException(GraphArray g, int vert);
		virtual const char* what() const throw();
	protected:
	private:
};

#endif // GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
