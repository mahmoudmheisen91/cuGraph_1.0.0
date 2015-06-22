#include <main/Graph.h>
#include <main/GraphDraw.h>
#include <QApplication>
#include <iostream>
#include <main/gstream.h>
#include <main/dataTypes.h>
#include <omp.h>

using namespace std;
using namespace cuGraph;

int main(int argc, char** argv) {
    /*Graph g0;
    g0.setType(UN_DIRECTED, SELF_LOOP);
    g0.setNumberOfVertices(10000);

    double t1 = omp_get_wtime();
        g0.fillByPER(100000000, 0.5);
    t1 = omp_get_wtime() - t1;

    cout << g0.getNumberOfEdges() << " , time = " << t1 << endl;
    cout << g0.countEdges() << endl << endl;*/

    Graph g1;
    g1.setType(UN_DIRECTED, SELF_LOOP);
    g1.setNumberOfVertices(10000);

    double t1 = omp_get_wtime();
        g1.fillByPZER(100000000, 0.5, 3);
    t1 = omp_get_wtime() - t1;

    cout << g1.getNumberOfEdges() << " , time = " << t1 << endl;
    cout << g1.countEdges() << endl << endl;

    /*Graph g3;
    g3.setType(UN_DIRECTED, SELF_LOOP);
    g3.setNumberOfVertices(10000);

    t1 = omp_get_wtime();
        g3.fillByPPreZER(100000000, 0.5, 3, 8);
    t1 = omp_get_wtime() - t1;

    cout << g3.getNumberOfEdges() << " , time = " << t1 << endl;
    cout << g3.countEdges() << endl << endl;

    Graph g5;
    g5.setType(UN_DIRECTED, SELF_LOOP);
    g5.setNumberOfVertices(10000);

    t1 = omp_get_wtime();
        g5.fillByBaselineER(100000000, 0.5);
    t1 = omp_get_wtime() - t1;

    cout << g5.getNumberOfEdges() << " , time = " << t1 << endl;
    cout << g5.countEdges() << endl << endl;

    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);
    g2.setNumberOfVertices(10000);

    t1 = omp_get_wtime();
        g2.fillByZER(100000000, 0.5);
    t1 = omp_get_wtime() - t1;

    cout << g2.getNumberOfEdges() << " , time = " << t1 << endl;
    cout << g2.countEdges() << endl << endl;

    Graph g4;
    g4.setType(UN_DIRECTED, SELF_LOOP);
    g4.setNumberOfVertices(10000);

    t1 = omp_get_wtime();
        g4.fillByPreZER(100000000, 0.5, 8);
    t1 = omp_get_wtime() - t1;

    cout << g4.getNumberOfEdges() << " , time = " << t1 << endl;
    cout << g4.countEdges() << endl << endl;

#ifdef DEBUG
    cout << "Debug" << endl;
#else
    cout << "Release" << endl;
#endif

//    GraphDraw draw(argc, argv);
//    draw.setGraph(&g1);
//    draw.randomPositions();
//    draw.run();
The gml format is fine. Since we have a large collection of graphs in
the .mtx-format (matrix market format), I would like to see this format
supported as well. The format is quite easy. Basically it starts with
a line containing 3 numbers (for regular graphs: the number of nodes
is listed twice followed by the number of edges). Each of the subsequent
lines in the text file defines one edge of the graph. A line consists
of two numbers: the source node number and the target node number
(both in the range [1..num-nodes]).

*/

    ogstream gos;
    gos.open("output/MTX/test.mtx");
    gos << g1;
    gos.close();

    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);

    igstream gis;
    gis.open("output/MTX/test.mtx");
    gis >> g2;
    gis.close();
    cout << g2.countEdges() << " " << g2.getNumberOfVertices() << endl;

    /*igstream gis;
    gis.open("src/gmls/test.txt");
    Graph g2;
    g2.setType(UN_DIRECTED, SELF_LOOP);
    g2.setNumberOfVertices(10);
    gis >> g2;
    cout << g2.isEmpty() << endl;
    gis.close();*/

    return 0;
}









