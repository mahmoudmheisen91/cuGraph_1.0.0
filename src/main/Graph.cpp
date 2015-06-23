#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>

#include <main/Path.h>
#include <main/Graph.h>
#include <main/dataTypes.h>
#include <main/Exceptions.h>
#include <cuda/Parallel_functions.h>

using namespace std;

namespace cuGraph {

    Graph::Graph() :numberOfVertices(0) {
        size = 0;
        numberOfEdges = 0;
        content = NULL;

        // Default values:
        direction = UN_DIRECTED;
        loop = SELF_LOOP;

        isInit = false;
        initDevice();
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

        isInit = true;
        initDevice();
    }

    Graph::~Graph(void) {
        delete content;
    }

    void Graph::setType(int dir, int lp) {
        clear();

        checkDir(dir);
        checkLoop(lp);

        direction = dir;
        loop = lp;
    }

    void Graph::setNumberOfVertices(int verts) {

        checkVertixesBound(verts);
        isInit = true;

        numberOfVertices = verts;
        size = pow(numberOfVertices, 2);

        clear();
    }

    void Graph::setNumberOfEdges(int edges) {
        checkEdgesBound(edges);
        numberOfEdges = edges;
    }

    void Graph::clear(void) {
        if(content != NULL)
            delete content;

        numberOfEdges = 0;
        content = new bool[size];
        fill(content, content+size, 0);
    }

    void Graph::addEdge(int v1, int v2) {

        if(numberOfVertices > 0)
            isInit = true;

        if(isFull())
            throw new GraphIsFullException();

        checkVertixName(v1, v2);

        if (!isDirectlyConnected(v1, v2)) {
            content[v1 * numberOfVertices + v2] = true;

            if (direction == UN_DIRECTED)
                content[v2 * numberOfVertices + v1] = true;
            numberOfEdges++;
        }
    }

    void Graph::removeEdge(int v1, int v2) {

        if(numberOfVertices > 0)
            isInit = true;

        if(isEmpty())
            throw new GraphIsEmptyException();

        checkVertixName(v1, v2);

        if(isDirectlyConnected(v1, v2)) {
            content[v1 * numberOfVertices + v2] = false;

            if (direction == UN_DIRECTED)
                content[v2 * numberOfVertices + v1] = false;
            numberOfEdges--;
        }
    }

    // from edge prespective not vertix (because vertices is constant (cuda))
    bool Graph::isFull(void) {

        if(isInit) {
            if(numberOfEdges == pow(numberOfVertices, 2))
                return true;
            else
                return false;
        }

        else
            throw new GraphIsNotInitException();
    }

    // from edge prespective not vertix (because vertices is constant (cuda))
    bool Graph::isEmpty(void) {
        if(isInit) {
            if(numberOfEdges == 0)
                return true;
            else
                return false;
        }
        else
            throw new GraphIsNotInitException();
    }

    bool Graph::isConnected(int v1, int v2) {
        checkVertixName(v1, v2);

        Path p(this, v1);
        return p.hasPathTo(v2);
    }

    bool Graph::isDirectlyConnected(int v1, int v2) {
        checkVertixName(v1, v2);

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

    void Graph::fillByPER(int E, double p) {
        checkEdgesBound(E);

        //parallel_PER(content, p, numberOfVertices, E);
    }

    void Graph::fillByPZER(int E, double p, int lambda) {
        checkEdgesBound(E);

        parallel_PZER(content, p, lambda, numberOfVertices, E);
    }

    void Graph::fillByPPreZER(int E, double p, int lambda, int m) {
        checkEdgesBound(E);

        //parallel_PPreZER(content, p, lambda, m, numberOfVertices, E);
    }

    int Graph::getNumberOfVertices(void) {
        return numberOfVertices;
    }

    int Graph::getNumberOfEdges(void) {
        return numberOfEdges;
    }

    int Graph::getDirection(void) {
        return direction;
    }

    int Graph::getLoop(void) {
        return loop;
    }

    int Graph::countEdges(void) {
        int E = 0;
        for(int i = 0; i < numberOfVertices; i++) {
            for(int j = 0; j < numberOfVertices; j++) {
                if(isDirectlyConnected(i, j))
                    E++;
            }
        }

        if(getDirection() == DIRECTED) {
            numberOfEdges = E;
            return E;
        }

        numberOfEdges = E/2;
        return E/2;
    }

    void Graph::checkDir(int dir) {
        if(dir != UN_DIRECTED && dir != DIRECTED)
            throw new GraphDirectionTypeException();
    }

    void Graph::checkLoop(int lp) {
        if(lp != SELF_LOOP && lp != NO_SELF_LOOP)
            throw new GraphLoopTypeException();
    }

    void Graph::checkVertixName(int v1, int v2) {
        if (v1 < 0 || v1 >= numberOfVertices)
            throw new GraphVertexOutOfBoundsException(numberOfVertices, v1);

        if (v2 < 0 || v2 >= numberOfVertices)
            throw new GraphVertexOutOfBoundsException(numberOfVertices, v2);

        if (v1 == v2 && loop == NO_SELF_LOOP)
            throw new GraphLoopTypeException();
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
