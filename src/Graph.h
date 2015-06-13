#ifndef GRAPH_H_
#define GRAPH_H_
// acycle graph

#include <iostream>
#include <string>
#include "dataTypes.h"

using namespace std;

namespace cuGraph {

    class Graph {
        friend class GraphDraw;
        friend class Path;
        friend class ogstream;
        friend class igstream;

        public:
            Graph();
            Graph(int numberOfVertices);
            virtual ~Graph();

            void setType(int dir, int lp);
            void setNumberOfVertices(int verts);

            void clear(void);
            void addEdge(int v1, int v2);
            void removeEdge(int v1, int v2);

            bool isEmpty(void);
            bool isConnected(int v1, int v2);
            bool isFullyConnected(void);
            bool isDirectlyConnected(int v1, int v2);

            void fillByBaselineER(int E, double p);
            void fillByZER(int E, double p);
            void fillByPreLogZER(int E, double p);
            void fillByPreZER(int E, double p, int m);
            void fillByPER(int E, double p);
            void fillByPZER(int E, double p);
            void fillByPPreZER(int E, double p, int m);

        protected:
            int  size;
            int* content;
            int  numberOfVertices;
            int  numberOfEdges;
            int direction;
            int loop;

        private:
            void checkVertixName(int vert);
            void checkEdgesBound(int edge);
            void checkVertixesBound(int verts);
        };
}
#endif // GRAPH_H_













