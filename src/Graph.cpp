#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Path.h"
#include "Graph.h"
#include "Exceptions.h"
#include <stdio.h>
#include <fstream>
#include <assert.h>
#include <string>
#include <cstdlib>
#include <cstring>

using namespace std;

namespace cuGraph {

    Graph::Graph() :numberOfVertices(0) {
        size = 0;
        numberOfEdges = 0;
        content = NULL;

        // Default values:
        direction = UN_DIRECTED;
        loop = SELF_LOOP;
    }

    Graph::Graph(int V) :numberOfVertices(V) {
        checkVertixesBound(V);
        size = pow(numberOfVertices, 2);
        numberOfEdges = 0;
        content = new bool[size];

        fill(content, content+size, 0);

        // Default values:
        direction = UN_DIRECTED;
        loop = SELF_LOOP;
    }

    Graph::~Graph(void) {
        delete content;
    }

    void Graph::setType(int dir, int lp) {
        if(!isEmpty())
            clear();

        direction = dir;
        loop = lp;
    }

    void Graph::setNumberOfVertices(int verts) {
        checkVertixesBound(verts);

        numberOfVertices = verts;
        size = pow(numberOfVertices, 2);

        clear();
    }

    void Graph::clear(void) {
        if(content != NULL)
            delete content;

        numberOfEdges = 0;
        content = new bool[size];
        fill(content, content+size, 0);
    }

    void Graph::addEdge(int v1, int v2) {
        if(isFull())
            throw new GraphIsFullException();

        checkVertixName(v1);
        checkVertixName(v2);
        content[v1 * numberOfVertices + v2] = true;
        numberOfEdges++;

        if (direction == UN_DIRECTED)
            content[v2 * numberOfVertices + v1] = true;
    }

    void Graph::removeEdge(int v1, int v2) {
        if(isEmpty())
            throw new GraphIsEmptyException();

        checkVertixName(v1);
        checkVertixName(v2);

        if(isDirectlyConnected(v1, v2)) {
            content[v1 * numberOfVertices + v2] = false;

            if (direction == UN_DIRECTED)
                content[v2 * numberOfVertices + v1] = false;
            numberOfEdges--;
        }
    }

    // from edge prespective not vertix (because vertices is constant (cuda))
    bool Graph::isFull(void) {
        if(numberOfEdges == pow(numberOfVertices, 2))
            return true;

        return false;
    }

    // from edge prespective not vertix (because vertices is constant (cuda))
    bool Graph::isEmpty(void) {
        if(numberOfEdges == 0)
            return true;

        return false;
    }

    bool Graph::isConnected(int v1, int v2) {
        checkVertixName(v1);
        checkVertixName(v2);

        Path p(this, v1);
        return p.hasPathTo(v2);
    }

    bool Graph::isDirectlyConnected(int v1, int v2) {
        checkVertixName(v1);
        checkVertixName(v2);

        if (direction == UN_DIRECTED) {
            if (content[v1 * numberOfVertices + v2] && content[v2 * numberOfVertices + v1])
                return true;
        } else {
            if (content[v1 * numberOfVertices + v2])
                return true;
        }
        return false;
    }

    void Graph::fillByBaselineER(int E, double p) {// TODO: check p
        checkEdgesBound(E);
        srand(time(0));
        double theta;

        int v1, v2;
        for(int i = 0; i < E; i++) {
            theta = (double)rand() / RAND_MAX;

            if (theta < p) {
                v1 = i / numberOfVertices;
                v2 = i % numberOfVertices;
                if(v1 != v2) {
                    addEdge(v1, v2);
                }
                else {
                    if(loop == SELF_LOOP)
                        addEdge(v1, v2);
                }
            }
        }
    }

    void Graph::fillByZER(int E, double p) {
        checkEdgesBound(E);
        srand(time(0));
        double theta, logp;

        int v1, v2, k, i = -1;
        while (i < E) {
            theta = (double)rand() / RAND_MAX;
            logp = log10f(theta)/log10f(1-p);

            k = max(0, (int)ceil(logp) - 1);
            i += k + 1;

            if(i < E) { // equavelent to: Discard last edge, because i > E
                v1 = i / numberOfVertices;
                v2 = i % numberOfVertices;
                if(v1 != v2) {
                    addEdge(v1, v2);
                }
                else {
                    if(loop == SELF_LOOP)
                        addEdge(v1, v2);
                }
            }
        }
    }

    void Graph::fillByPreLogZER(int E, double p) {
        checkEdgesBound(E);
        srand(time(0));
        double *logp = new double[RAND_MAX];
        double c;

        c = log10f(1-p);

        for(int i = 0; i < RAND_MAX; i++) {
            logp[i] = log10f(i/ RAND_MAX);
        }

        int theta, v1, v2, k, i = -1;
        while (i < E) {
            theta = rand();

            k = max(0, (int)ceil(logp[theta] / c) - 1);
            i += k + 1;

            if(i < E) { // equavelent to: Discard last edge, because i > E
                v1 = i / numberOfVertices;
                v2 = i % numberOfVertices;
                if(v1 != v2) {
                    addEdge(v1, v2);
                }
                else {
                    if(loop == SELF_LOOP)
                        addEdge(v1, v2);
                }
            }
        }
    }

    void Graph::fillByPreZER(int E, double p, int m) {
        checkEdgesBound(E);
        srand(time(0));
        double theta, logp;
        double *F = new double[m+1];

        for(int i = 0; i <= m; i++) {
            F[i] = 1 - pow(1-p, i+1);
        }

        int v1, v2, k, j, i = -1;
        while(i < E) {
            theta = (double)rand() / RAND_MAX;

            j = 0;
            while(j <= m) {
                if(F[j] > theta) {
                    k = j;
                    break; // must break from j while loop not i while loop
                }
                j++;
            }

            // if could not find k from the upper loop
            if(j == m+1) { // rare to happen for large m value
                logp = log10f(1-theta)/log10f(1-p);
                k = max(0, (int)ceil(logp) - 1);
            }

            i += k + 1;

            if(i < E) { // equavelent to: Discard last edge, because i > E
                v1 = i / numberOfVertices;
                v2 = i % numberOfVertices;
                if(v1 != v2) {
                    addEdge(v1, v2);
                }
                else {
                    if(loop == SELF_LOOP)
                        addEdge(v1, v2);
                }
            }
        }
    }

    void Graph::checkVertixName(int vert) {
        if (vert < 0 || vert >= numberOfVertices)
            throw new GraphVertexOutOfBoundsException(numberOfVertices, vert);
    }

    void Graph::checkEdgesBound(int edge) {
        if (edge < 0 || edge > size)
            throw new GraphEdgeOutOfBoundsException(size, edge);
    }

    void Graph::checkVertixesBound(int verts) {
        if (verts <= 0)
            throw new GraphNumberOfVertexOutOfBoundsException(verts);
    }
}
