#ifndef ORIGINALERGRAPH_H_
#define ORIGINALERGRAPH_H_

#include "GraphArray.h"


class OriginalERGraph {
	public:
		OriginalERGraph(int V, double p);
		virtual ~OriginalERGraph();
		int getNumberOfVertices();
		int getNumberOfEdges();
		int getSize();
		void printGraphAsArray(void);
		void addEdge(int v1, int v2);
		void removeEdge(int v1, int v2);
		bool isDirectlyConnected(int v1, int v2);
		bool isConnected(int v1, int v2);
	protected:
	private:
		GraphArray *graph;
};

#endif // ORIGINALERGRAPH_H_
