#include "Graph.h"
#include "GraphDraw.h"
#include "Editor.h"
#include <QApplication>
#include <iostream>

using namespace std;
using namespace cuGraph;

int main(int argc, char** argv) {
    Graph g1(10);
    g1.fillByBaselineER(100, 0.5);

    int a = 10;
    a = a+1;
    std::cout << a << std::endl;

    GraphDraw draw(argc, argv);
    draw.setGraph(&g1);
    draw.randomPositions();
    draw.exec();
}









