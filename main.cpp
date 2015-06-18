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
    double t1 = omp_get_wtime();

    Graph g1;
    g1.setType(UN_DIRECTED, SELF_LOOP);
    g1.setNumberOfVertices(10000);
    g1.fillByPZER(40000000, 0.5, 3);
    t1 = omp_get_wtime() - t1;
    cout << "time = " << t1 << endl;

#ifdef DEBUG
    cout << "Debug" << endl;
#else
    cout << "Release" << endl;
#endif

    t1 = omp_get_wtime();

    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);
    g2.setNumberOfVertices(10000);
    g2.fillByZER(40000000, 0.5);
    t1 = omp_get_wtime() - t1;
    cout << "time = " << t1 << endl;

    Graph g3;
    g3.setType(UN_DIRECTED, SELF_LOOP);
    g3.setNumberOfVertices(10);
    g3.fillByZER(10, 0.5);

    GraphDraw draw(argc, argv);
    draw.setGraph(&g3);
    draw.randomPositions();
    draw.run();



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









