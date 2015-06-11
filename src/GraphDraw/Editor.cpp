Skip to content
This repository
Pull requests
Issues
Gist
 @mahmoudmheisen91
 Unwatch 1
  Star 0
  Fork 0
mahmoudmheisen91/cuGraph_1.0.0
 tree: 13d9141851  cuGraph_1.0.0/src/GraphDraw/Editor.cpp
@mahmoudmheisen91mahmoudmheisen91 an hour ago Add arrow line method to the Editor class
1 contributor
RawBlameHistory    165 lines (128 sloc)  4.573 kB
#include "Editor.h"
#include <QPainter>
#include <QApplication>
#include <QPixmap>
#include <QWidget>
#include <QThread>
#include <math.h>
#include <iostream>

using namespace std;

namespace cuGraph {

    Editor::Editor() : QWidget(0) {
        setFixedSize(1300, 650);
        move(60, 60);
        setWindowTitle("cuGraph - draw");
        QWidget::show();

        color.set(0, 0, 0);
        lineWidth = 1;
        pen = new QPen;
        map = new QPixmap(1300, 650);
        (*map).fill();
    }

    Editor::~Editor() {
        delete map;
        delete pen;
    }

    void Editor::line(Point p0, Point p1) {
        QThread::yieldCurrentThread();

        (*pen).setWidth(lineWidth);
        (*pen).setBrush(QColor(color.red, color.green, color.blue, 255));

        QPainter *painter = new QPainter(map);
        (*painter).setRenderHint(QPainter::Antialiasing, true);
        (*painter).setPen(*pen);

        (*painter).drawLine(QPointF(p0.x, p0.y), QPointF(p1.x, p1.y));
        delete painter;
    }

    void Editor::arrowLine(Point p0, Point p1) {
        line(p0, p1);
        arrow(p0, p1);
    }


    void Editor::doubleArrowLine(Point p0, Point p1) {
        arrow(p0, p1);
        line(p0, p1);
        arrow(p1, p0);
    }

    void Editor::circle(Point p, double rad) {
        QThread::yieldCurrentThread();

        (*pen).setWidth(3);
        (*pen).setColor(QColor(0, 0, 0, 255));

        QBrush filling;
        filling.setStyle(Qt::SolidPattern);
        filling.setColor(QColor(color.red, color.green, color.blue, 255));

        QPainter *painter = new QPainter(map);
        (*painter).setRenderHint(QPainter::Antialiasing, true);
        (*painter).setBrush(filling);
        (*painter).setPen(*pen);
        (*painter).drawEllipse(QPointF(p.x, p.y), rad, rad);
        delete painter;
    }

    void Editor::text(Point p, QString text) {
        QThread::yieldCurrentThread();

        (*pen).setBrush(QColor(color.red, color.green, color.blue, 255));

        QPainter *painter = new QPainter(map);
        (*painter).setRenderHint(QPainter::Antialiasing, true);
        (*painter).setFont(QFont("Arial", 12));
        (*painter).setPen(*pen);
        (*painter).drawText(QPointF(p.x+5, p.y+15), text);
        delete painter;
    }

    void Editor::setLineWidth(double width) {
        QThread::yieldCurrentThread();
        lineWidth = width;
    }

    void Editor::setColor(Color newColor) {
        QThread::yieldCurrentThread();
        color.set(newColor);
    }

    void Editor::save(QString filename) {
        QThread::yieldCurrentThread();
        map->save(filename);
    }

    void Editor::paintEvent(QPaintEvent *) {
        QPainter(this).drawPixmap(0, 0, *map);
    }

    Point Editor::midPoint(Point p0, Point p1) {
        return Point((p0.x+p1.x)/2, (p0.y+p1.y)/2);
    }

    void Editor::arrow(Point p0, Point p1) {
        Point mid;
        mid.set(midPoint(p0, p1));

        double width = lineWidth + 2;
        double height= 15;
        double x0,x1,x2,x3,x4,x5,x6,x7,y0,y1,y2,y3,y4,y5,y6,y7;
        double distance = sqrt(pow(mid.x - p0.x, 2) + pow(mid.y - p0.y, 2));
        double dx = mid.x + (p0.x - mid.x) * height / distance;
        double dy = mid.y + (p0.y - mid.y) * height / distance;
        double k = width / height;

        x0 = dx - (dy - mid.y) * k;
        y0 = dy - (mid.x - dx) * k;
        x1 = mid.x;
        y1 = mid.y;
        x2 = (dy - mid.y) * k + dx;
        y2 = (mid.x - dx) * k + dy;
        x7 = (x0+x2) / 2.0;
        y7 = (y0+y2) / 2.0;

        k = (lineWidth/2) / distance;

        x3 = (p0.y - y7) * k + p0.x;
        y3 = (x7 - p0.x) * k + p0.y;
        x4 = p0.x - (p0.y - y7) * k;
        y4 = p0.y - (x7 - p0.x) * k;
        x5 = (y7 - p0.y) * k + x7;
        y5 = (p0.x - x7) * k + y7;
        x6 = x7 - (y7 - p0.y) * k;
        y6 = y7 - (p0.x - x7) * k;

        QPointF points[7] = {QPointF(x0,y0), QPointF(x1,y1), QPointF(x2,y2), QPointF(x6,y6),
                            QPointF(x3,y3), QPointF(x4,y4), QPointF(x5,y5)};

        (*pen).setWidth(0);
        (*pen).setBrush(QColor(color.red, color.green, color.blue, 255));

        QBrush filling;
        filling.setStyle(Qt::SolidPattern);
        filling.setColor(QColor(color.red, color.green, color.blue, 255));

        QPainter *painter = new QPainter(map);
        (*painter).setPen(*pen);
        (*painter).setBrush(filling);
        (*painter).setRenderHint(QPainter::Antialiasing, true);
        (*painter).drawPolygon(points,7);
        delete painter;
    }

} // end of namespace


Status API Training Shop Blog About
© 2015 GitHub, Inc. Terms Privacy Security Contact
