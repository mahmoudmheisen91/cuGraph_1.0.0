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
    editor.setLineWidth(5);
    editor.setColor(Color(127, 127, 127));
    editor.line(Point(0, 0), Point(600, 600));
    editor.save("testImage.png");

    int a = 10;
    a = a+1;
    std::cout << a << std::endl;

    app.exec(); // wait until window is closed
    return retcode; // after finishing pass result to main
}
