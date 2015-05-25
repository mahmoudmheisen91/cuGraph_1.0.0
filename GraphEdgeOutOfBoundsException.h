#ifndef GRAPHEDGEOUTOFBOUNDSEXCEPTION_H_
#define GRAPHEDGEOUTOFBOUNDSEXCEPTION_H_

#include <exception>
#include <string>
using namespace std;

class GraphEdgeOutOfBoundsException : public exception
{
	public:
		GraphEdgeOutOfBoundsException(int size, int edge);
		virtual string what(int size, int vert) const throw();
	protected:
	private:
};

#endif // GRAPHEDGEOUTOFBOUNDSEXCEPTION_H_
