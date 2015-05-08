#ifndef GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
#define GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_

#include <exception>
using namespace std;

class GraphVertexOutOfBoundsException: public exception
{
	public:
		GraphVertexOutOfBoundsException();
		virtual ~GraphVertexOutOfBoundsException();
		virtual const char* what() const throw();
	protected:
	private:
};

#endif // GRAPHVERTEXOUTOFBOUNDSEXCEPTION_H_
class myexception: public exception {

};

int main () {
  try
  {
    throw myex;
  }
  catch (exception& e)
  {
    cout << e.what() << '\n';
  }
  return 0;
}
