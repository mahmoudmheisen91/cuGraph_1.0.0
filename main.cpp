#include <omp.h>
#include <iostream>
#include <main/cuGraph.h>

using namespace std;
using namespace cuGraph;

int main(int argc, char** argv) {

    Graph g1;
    g1.setType(DIRECTED);
    g1.setNumberOfVertices(10000);

    double t1 = omp_get_wtime();
        g1.fillByPZER(100000000, 0.5, 3);
    t1 = omp_get_wtime() - t1;

    cout << endl << "Time = " << t1 << ", Edge count = " << g1.countEdges() << endl << endl;
/*
    ogstream ogs;
    ogs.open("output/MTX/test.mtx");
    ogs << g1;
    ogs.close();

    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);

    igstream igs;
    igs.open("output/MTX/test.mtx");
    igs >> g2;
    igs.close();
    cout << "Edge count = " << g2.countEdges();
    cout << ", Vertix count = " << g2.getNumberOfVertices() << endl;
*/
    return 0;
}









