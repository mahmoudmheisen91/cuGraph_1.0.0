#ifndef GRAPHDRAW_H
#define GRAPHDRAW_H

#include "Editor.h"
#include "Graph.h"

namespace cuGraph {

    class GraphDraw {

        public:
            GraphDraw(int argc, char **argv);
            virtual ~GraphDraw();

            void setGraph(Graph *g);
            void setDrawSettings(Settings set);

            void run(void);
            void randomPositions(void);

        protected:

        private:
            double *posx, *posy;
            Settings draw_settings;
            int verts, edges;
            QApplication *app;
            Editor *draw_editor;
            Graph *g_global;
    };
}

#endif // GRAPHDRAW_H