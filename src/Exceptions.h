#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <exception>
#include <string>
using namespace std;

class GraphVertexOutOfBoundsException: public exception {
    public:
        GraphVertexOutOfBoundsException(int size, int vert);
        virtual string what(int size, int vert) const throw();
    protected:
    private:
};

class GraphEdgeOutOfBoundsException : public exception {
    public:
        GraphEdgeOutOfBoundsException(int size, int edge);
        virtual string what(int size, int vert) const throw();
    protected:
    private:
};

class GraphNumberOfVertexOutOfBoundsException : public exception {
    public:
        GraphNumberOfVertexOutOfBoundsException(int verts);
        virtual string what(int verts) const throw();
    protected:
    private:
};

#endif // EXCEPTIONS_H_
