#ifndef EDITOR_H_
#define EDITOR_H_

#include "dataTypes.h"
#include <QPixmap>
#include <QWidget>
#include <QString>
#include <QPaintEvent>

namespace cuGraph {

    class Editor : public QWidget { Q_OBJECT
        public:
            Editor();
            virtual ~Editor();
            void line(Point p0, Point p1);
            void arrowLine(Point p0, Point p1);
            void doubleArrowLine(Point p0, Point p1);
            void circle(Point p, double rad);
            void text(Point p, QString text);

            void setLineWidth(double width);
            void setColor(Color newColor);
            void save(QString filename);
            double getHeight();
            double getWidth();

        protected:
            void paintEvent(QPaintEvent *);

        private:
            QPen *pen;
            QPixmap* map;
            Color color;
            double lineWidth;
            double height, width;
            Point midPoint(Point p0, Point p1);
            void arrow(Point p0, Point p1);
    };
}

#endif
