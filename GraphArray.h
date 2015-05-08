#ifndef GRAPHARRAY_H_
#define GRAPHARRAY_H_

class GraphArray {
	public:
		GraphArray(int numberOfVertices);
		virtual ~GraphArray();
		int getNumberOfVertices();
		void printGraphAsArray(void);
		void addEdge(int v1, int v2);
		bool isConnected(int v1, int v2);
	protected:
	private:
		int numberOfVertices;
		int size;
		int* content;
		void checkVertixName(int vert);
};

#endif // GRAPHARRAY_H_
