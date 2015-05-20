#ifndef GRAPHARRAY_H_
#define GRAPHARRAY_H_

class GraphArray {
	public:
		GraphArray(int numberOfVertices);
		virtual ~GraphArray();
		int getNumberOfVertices();
		int getNumberOfEdges();
		int getSize();
		void printGraphAsArray(void);
		void addEdge(int v1, int v2);
		void removeEdge(int v1, int v2);
		bool isDirectlyConnected(int v1, int v2);
		bool isConnected(int v1, int v2);
		int* content;
	protected:
	private:
		int size;
		int numberOfVertices;
		int numberOfEdges;
		void checkVertixName(int vert);
};

#endif // GRAPHARRAY_H_
