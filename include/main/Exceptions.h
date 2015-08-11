/*
 * Exceptions.h
 *
 *  Created: 2015-05-24, Modified: 2015-08-11
 *
 */
#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

// Standard C++ libraries includes:
#include <exception>
#include <string>

// Namespaces:
using namespace std;

namespace cuGraph {
	// Throw this exception, if the vertix out of bound (0 - max-1):
    class GraphVertexOutOfBoundsException: public exception {
        public:
            GraphVertexOutOfBoundsException(long long int size, int vert);
            virtual string what(long long int size, int vert) const throw();
        protected:
        private:
    };

	// Throw this exception, if the edge count exceeded max size:
    class GraphEdgeOutOfBoundsException : public exception {
        public:
            GraphEdgeOutOfBoundsException(long long int size, int edge);
            virtual string what(long long int size, int vert) const throw();
        protected:
        private:
    };

	// Throw this exception, if the vertix size exceeded max size:
    class GraphNumberOfVertexOutOfBoundsException : public exception {
        public:
            GraphNumberOfVertexOutOfBoundsException(long long int verts);
            virtual string what(long long int verts) const throw();
        protected:
        private:
    };

	// Throw this exception, if the graph is full:
    class GraphIsFullException: public exception {
        public:
            GraphIsFullException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

	// Throw this exception, if the graph is empty:
    class GraphIsEmptyException: public exception {
        public:
            GraphIsEmptyException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

	// Throw this exception, if the direction is undefined:
    class GraphDirectionTypeException: public exception {
        public:
            GraphDirectionTypeException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

	// Throw this exception, if the graph is not initalized:
    class GraphIsNotInitException: public exception {
        public:
            GraphIsNotInitException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };
}

#endif // EXCEPTIONS_H_
