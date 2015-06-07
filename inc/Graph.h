#ifndef GRAPH_H_
#define GRAPH_H_
// acycle graph

#include "draw.h"

typedef struct _settings {
    double rangeMin;
    double rangeMax;
    Color color;
    double penWidth;
    double transparency;
    int fontSize;
    int windowWidth;
    int windowHeight;
} Settings;

class Graph {
	public:
        Graph();
        Graph(int numberOfVertices);
        virtual ~Graph();

        void setType(int dir, int lp);
        void setDrawSettings(Settings sets);

        void draw(void);

        void clear(void);
		void addEdge(int v1, int v2);
		void removeEdge(int v1, int v2);
        void printGraphAsArray(void);

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
        Settings defaultSettings;

		void checkVertixName(int vert);
		void checkEdgeRange(int edge);
};

double fa(double x, double k);
double fr(double x, double k);
double length(double array[2]);
int cool(int t);

#endif // GRAPH_H_


#define DIRECTED        0
#define UN_DIRECTED     1
#define SELF_LOOP       2 // TODO: conect loop with the generators
#define NO_SELF_LOOP    3













