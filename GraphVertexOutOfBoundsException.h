#ifndef GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
#define GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_

#include <exception>
using namespace std;

class GraphVertexOutOfBoundsException: public exception
{
	public:
		GraphVertexOutOfBoundsException();
		virtual const char* what() const throw();
	protected:
	private:
};

#endif // GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
