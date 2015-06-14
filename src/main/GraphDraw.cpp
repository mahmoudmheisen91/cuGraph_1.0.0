#include "GraphDraw.h"
#include <QApplication>
#include <iostream>
#include <math.h>

namespace cuGraph {

    GraphDraw::GraphDraw(int argc, char** argv) {
        app = new QApplication(argc, argv);
        draw_editor = new Editor;
        g_global = new Graph;

        draw_settings.node_color = Color(255, 0, 0);
        draw_settings.edge_color = Color(0, 0, 0);
        draw_settings.text_color = Color(0, 0, 0);
        draw_settings.line_width = 2;
        draw_settings.is_numbered = true;
        draw_settings.node_size = 15;
    }

    GraphDraw::~GraphDraw() {
        delete posx;
        delete posy;
        delete app;
        //delete g_global;    //TODO
        //delete draw_editor; //TODO
    }

    void GraphDraw::setGraph(Graph *g) {
        g_global = g;
        setGraphDrawType(g_global->direction, g_global->loop);
        verts = g_global->numberOfVertices;
        edges = g_global->numberOfEdges;

        posx = new double[verts];
        posy = new double[verts];
    }

    void GraphDraw::setGraphDrawType(int dir, int lp) {
        graph_direction = dir;
        graph_loop = lp;
    }

    void GraphDraw::run(void) {

        draw_editor->setColor(draw_settings.edge_color);
        draw_editor->setLineWidth(draw_settings.line_width);

        for (int v = 0; v < verts; v++) {
            for (int u = 0; u < verts; u++) {
                if (g_global->isDirectlyConnected(v, u)) {
                    if(graph_direction == UN_DIRECTED)
                        draw_editor->line(Point(posx[v], posy[v]), Point(posx[u], posy[u]));
                    else
                        draw_editor->arrowLine(Point(posx[v], posy[v]), Point(posx[u], posy[u]));
                }
            }
        }

        draw_editor->setColor(draw_settings.node_color);
        for (int v = 0; v < verts; v++) {
            draw_editor->circle(Point(posx[v], posy[v]), draw_settings.node_size);
        }

        if(draw_settings.is_numbered) {
            draw_editor->setColor(draw_settings.text_color);
            for (int v=0; v < verts; v++) {
                char str[10];
                sprintf(str, "%d", v);
                draw_editor->text(Point(posx[v], posy[v]), str);
            }
        }

        char str[100];
        sprintf(str, "Number of Vertices = %d, Number of Edges = %d", verts, edges);
        draw_editor->text(Point(190, 15), str);

        draw_editor->save("name.png");

        app->exec(); // wait until window is closed
    }

    void GraphDraw::randomPositions(void) {
        int W = draw_editor->getWidth() - 60;
        int L = draw_editor->getHeight() - 60;

        for (int i = 0; i < verts; i++) {
            posx[i] = 35 + (rand() / (double)RAND_MAX) * W;
            posy[i] = 35 + (rand() / (double)RAND_MAX) * L;
        }
    }

    void GraphDraw::setDrawSettings(Settings set) {
        draw_settings.node_color = set.node_color;
        draw_settings.edge_color = set.edge_color;
        draw_settings.line_width = set.line_width;
        draw_settings.is_numbered = set.is_numbered;
    }
}

