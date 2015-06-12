#ifndef GRAPHDRAW_H
#define GRAPHDRAW_H

#include "Editor.h"
#include "Graph.h"

namespace cuGraph {
    class GraphDraw {

        public:
            GraphDraw();
            GraphDraw(int argc, char **argv);
            virtual ~GraphDraw();
            void setGraphDrawType(int dir, int lp);
            void setGraph(Graph *g);
            void exec();
            void randomPositions();
            void setDrawSettings(Settings set);

        protected:

        private:
            double *posx, *posy;
            int graph_direction;
            int graph_loop;
            Settings draw_settings;
            int verts, edges;
            QApplication *app;
            Editor *draw_editor;
            Graph *g_global;
    };
}

#endif // GRAPHDRAW_H
