#ifndef GRAPH_H_
#define GRAPH_H_
// acycle graph

#define DIRECTED        0
#define UN_DIRECTED     1
#define SELF_LOOP       2 // TODO: conect loop with the generators
#define NO_SELF_LOOP    3

namespace cuGraph {

    class Graph {
        public:
            Graph();
            Graph(int numberOfVertices);
            virtual ~Graph();

            void setType(int dir, int lp);
            int getDir();
            int getLp();

            void clear(void);
            void addEdge(int v1, int v2);
            void removeEdge(int v1, int v2);
            void printGraphAsArray(void);
            int adjacentNodes(int v);

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

            int  getSize(void);
            int* getContent(void);
            int  getNumberOfEdges(void);
            int  getNumberOfVertices(void);

        protected:

        private:
            int  size;
            int* content;
            int  numberOfVertices;
            int  numberOfEdges;

            int direction;
            int loop;

            void checkVertixName(int vert);
            void checkEdgeRange(int edge);
        };
}
#endif // GRAPH_H_













