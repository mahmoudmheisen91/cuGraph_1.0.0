#include <main/Graph.h>
#include <main/GraphDraw.h>
#include <QApplication>
#include <iostream>
//#include "gstream.h"
#include <main/dataTypes.h>
#include <omp.h>

using namespace std;
using namespace cuGraph;

int main(int argc, char** argv) {
    Graph g1;
    g1.setType(UN_DIRECTED, SELF_LOOP);
    g1.setNumberOfVertices(10000);

    double t1 = omp_get_wtime();
        g1.fillByPZER(100000000, 0.5, 3);
    t1 = omp_get_wtime() - t1;

    cout << g1.getNumberOfEdges() << " , time = " << t1 << endl;

#ifdef DEBUG
    cout << "Debug" << endl;
#else
    cout << "Release" << endl;
#endif

    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);
    g2.setNumberOfVertices(10000);

    t1 = omp_get_wtime();
        g2.fillByZER(100000000, 0.5);
    t1 = omp_get_wtime() - t1;

    cout << g2.getNumberOfEdges() << " , time = " << t1 << endl;

//    GraphDraw draw(argc, argv);
//    draw.setGraph(&g1);
//    draw.randomPositions();
//    draw.run();



/*    ogstream gos;
    gos.open("src/gmls/test.txt");
    gos << g1;
    gos.close();

    igstream gis;
    gis.open("src/gmls/test.txt");
    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);
    g2.setNumberOfVertices(10);
    gis >> g2;
    cout << g2.isEmpty() << endl;
    gis.close();
*/
    return 0;
}









