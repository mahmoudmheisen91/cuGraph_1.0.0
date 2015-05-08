#ifndef GRAPHARRAY_H_
#define GRAPHARRAY_H_

class GraphArray {
	public:
		GraphArray(int numberOfVertices);
		//virtual ~GraphArray();
		int getNumberOfVertices();
		void setNumberOfVertices(int val);
		void printGraphAsArray(void);
	protected:
	private:
		int numberOfVertices;
		int size;
		int* content;
};

#endif // GRAPHARRAY_H_
