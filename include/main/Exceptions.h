#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <exception>
#include <string>

using namespace std;

namespace cuGraph {
    class GraphVertexOutOfBoundsException: public exception {
        public:
            GraphVertexOutOfBoundsException(long long int size, int vert);
            virtual string what(long long int size, int vert) const throw();
        protected:
        private:
    };

    class GraphEdgeOutOfBoundsException : public exception {
        public:
            GraphEdgeOutOfBoundsException(long long int size, int edge);
            virtual string what(long long int size, int vert) const throw();
        protected:
        private:
    };

    class GraphNumberOfVertexOutOfBoundsException : public exception {
        public:
            GraphNumberOfVertexOutOfBoundsException(long long int verts);
            virtual string what(long long int verts) const throw();
        protected:
        private:
    };

    class GraphIsFullException: public exception {
        public:
            GraphIsFullException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

    class GraphIsEmptyException: public exception {
        public:
            GraphIsEmptyException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

    class GraphDirectionTypeException: public exception {
        public:
            GraphDirectionTypeException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

    class GraphLoopTypeException: public exception {
        public:
            GraphLoopTypeException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };

    class GraphIsNotInitException: public exception {
        public:
            GraphIsNotInitException(void);
            virtual string what(int) const throw();
        protected:
        private:
    };
}

#endif // EXCEPTIONS_H_
