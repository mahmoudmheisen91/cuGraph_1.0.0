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
        this->r = 0; this->g = 0; this->b = 0;
        lineWidth = 1;
        this->setFixedSize(1300, 650);
        this->move(60, 60);
        this->setWindowTitle("Graph");
        this->QWidget::show();

        map = new QPixmap(1300, 650);
        (*map).fill();
    }

    void Editor::line(double x0, double y0, double x1, double y1) {
        QThread::yieldCurrentThread();

        QPen pen;
        pen.setStyle(Qt::SolidLine);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setWidth(lineWidth);
        pen.setBrush(QColor(r, g, b, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(pen);

        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1));
    }

    void Editor::arrow(double x0, double y0, double x1, double y1) {
        QThread::yieldCurrentThread();

        QPen pen;
        pen.setStyle(Qt::SolidLine);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setWidth(lineWidth);
        pen.setBrush(QColor(r, g, b, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(pen);

        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1));
    }

    void Editor::circle(double x, double y, double rr) {
        QThread::yieldCurrentThread();

        QPen border;
        border.setWidth(3);
        border.setColor(QColor(0, 0, 0, 255));

        QBrush filling;
        filling.setStyle(Qt::SolidPattern);
        filling.setColor(QColor(r, g, b, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setBrush(filling);
        painter.setPen(border);

        painter.drawEllipse(QPointF(x, y), rr, rr);
    }

    void Editor::text(QString text, double x, double y) {
        QThread::yieldCurrentThread();

        QPen pen;
        pen.setStyle(Qt::SolidLine);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setBrush(QColor(r, g, b, 255));

        QPainter painter(map);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setFont(QFont("Arial", 12));
        painter.setPen(pen);

        painter.drawText(QPointF(x+5, y+15), text);
    }

    void Editor::setLineWidth(double w) {
        QThread::yieldCurrentThread();

        this->lineWidth = w;
    }

    void Editor::setcolor(int rnew, int gnew, int bnew) {
        QThread::yieldCurrentThread();

        this->r = rnew;
        this->g = gnew;
        this->b = bnew;
    }

    void Editor::save(QString filename) {
        QThread::yieldCurrentThread();
        map->save(filename);
    }

    void Editor::paintEvent(QPaintEvent *) {
        QPainter(this).drawPixmap(0, 0, *map);
    }
} // end of 'draw' namespace
