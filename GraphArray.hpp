#ifndef GRAPHARRAY_HPP_
#define GRAPHARRAY_HPP_

#include <string>
using namespace std;

class GraphArray
{
	public:
		GraphArray(int numberOfVertices);
		virtual ~GraphArray();
		int getNumberOfVertices();
		void setNumberOfVertices(int val);
	protected:
	private:
		int numberOfVertices;
		int content[];
};

//void printMatrix(int a[][5], int row, int col) {
//    for(int i = 0; i < row; i++) {
//        for(int j = 0; j < col; j++) {
//            cout << a[i][j] << " ";
//        }

//        cout << endl;
//    }
//}

#endif // GRAPHARRAY_HPP_
