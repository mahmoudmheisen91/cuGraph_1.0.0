#ifndef EDITOR_H_
#define EDITOR_H_

#include <QPixmap>
#include <QWidget>
#include <QString>
#include <QPaintEvent>

namespace cuGraph {

    class Editor : public QWidget { Q_OBJECT
        public:
            Editor();
            void line(double x0, double y0, double x1, double y1);
            void arrow(double x0, double y0, double x1, double y1);
            void circle(double x, double y, double rr);
            void text(QString text, double x, double y);

            void setLineWidth(double w);
            void setcolor(int rnew, int gnew, int bnew);
            void save(QString filename);

        protected:
            void paintEvent(QPaintEvent *);

        private:
            QPixmap* map;
            int r, g, b;
            double lineWidth;
    };
}

#endif
