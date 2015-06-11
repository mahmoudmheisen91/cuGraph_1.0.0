#include "Editor.h"
#include <QPainter>
#include <QApplication>
#include <QPixmap>
#include <QWidget>
#include <QThread>
#include <iostream>

using namespace std;

namespace cuGraph {

    Editor::Editor() : QWidget(0) {
        color.setColor(0, 0, 0);
        lineWidth = 1;
        this->setFixedSize(1300, 650);
        this->move(60, 60);
        this->setWindowTitle("Graph");
        this->QWidget::show();

        map = new QPixmap(1300, 650);
        (*map).fill();
    }

    void Editor::line(Point p0, Point p1) {
        QThread::yieldCurrentThread();

        QPen pen;
        pen.setStyle(Qt::SolidLine);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setWidth(lineWidth);
        pen.setBrush(QColor(color.red, color.green, color.blue, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(pen);

        painter.drawLine(QPointF(p0.x, p0.y), QPointF(p1.x, p1.y));
    }

    void Editor::arrow(Point p0, Point p1) {
        QThread::yieldCurrentThread();

        QPen pen;
        pen.setStyle(Qt::SolidLine);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setWidth(lineWidth);
        pen.setBrush(QColor(color.red, color.green, color.blue, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(pen);

        painter.drawLine(QPointF(p0.x, p0.y), QPointF(p1.x, p1.y));
    }

    void Editor::circle(Point p, double rad) {
        QThread::yieldCurrentThread();

        QPen border;
        border.setWidth(3);
        border.setColor(QColor(0, 0, 0, 255));

        QBrush filling;
        filling.setStyle(Qt::SolidPattern);
        filling.setColor(QColor(color.red, color.green, color.blue, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setBrush(filling);
        painter.setPen(border);

        painter.drawEllipse(QPointF(p.x, p.y), rad, rad);
    }

    void Editor::text(Point p, QString text) {
        QThread::yieldCurrentThread();

        QPen pen;
        pen.setStyle(Qt::SolidLine);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setBrush(QColor(color.red, color.green, color.blue, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setFont(QFont("Arial", 12));
        painter.setPen(pen);

        painter.drawText(QPointF(p.x+5, p.y+15), text);
    }

    void Editor::setLineWidth(double w) {
        QThread::yieldCurrentThread();

        this->lineWidth = w;
    }

    void Editor::setColor(Color newColor) {
        QThread::yieldCurrentThread();

        color.setColor(newColor.red, newColor.green, newColor.blue);
    }

    void Editor::save(QString filename) {
        QThread::yieldCurrentThread();
        map->save(filename);
    }

    void Editor::paintEvent(QPaintEvent *) {
        QPainter(this).drawPixmap(0, 0, *map);
    }
} // end of 'draw' namespace
