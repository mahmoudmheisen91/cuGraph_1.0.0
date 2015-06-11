#include "Editor.h"
#include <QApplication>
#include <iostream>

using namespace std;
using namespace cuGraph;

int main(int argc, char** argv) {
    QAtomicInt retcode(0);
    QApplication app(argc, argv);

    Editor editor;
    editor.setColor(Color(0, 127, 0));
    editor.text(Point(0, 0), "Mahmoud Nidal Ibrahim Mheisen");
    editor.circle(Point(600, 300),20);
    editor.setLineWidth(4);
    editor.doubleArrowLine(Point(200, 300), Point(250, 60));

    editor.setColor(Color(255, 0, 0));
    editor.circle(Point(1000, 300),10);

    editor.setColor(Color(0, 0, 127));
    editor.line(Point(200, 400), Point(600, 500));

    editor.setColor(Color(80, 60, 50));
    editor.arrowLine(Point(800, 50), Point(1100, 300));

    editor.save("testImage.png");

    int a = 10;
    a = a+1;
    std::cout << a << std::endl;

    app.exec(); // wait until window is closed
    return retcode; // after finishing pass result to main
}
