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

#include <string>
using namespace std;

namespace cuGraph {

    Graph::Graph() {}

    Graph::Graph(int V) :numberOfVertices(V) {
        size = pow(numberOfVertices, 2);
        numberOfEdges = 0;
        content = new int[size];

        fill(content, content+size, 0);

        // Default values:
        direction = UN_DIRECTED;
        loop = SELF_LOOP;
    }

    Graph::~Graph(void) {
        delete content;
    }

    void Graph::readText(char *file_name) {
        ifstream myfile;
        myfile.open(file_name);

        myfile >> numberOfVertices;
        myfile >> numberOfEdges;

        int v1, v2;
        while (myfile >> v1) {
            myfile >> v2;
            content[v1 * numberOfVertices + v2] = 1;
        }
        myfile.close();
    }

    void Graph::writeText(char* file_name) {
        ofstream myfile;
        myfile.open(file_name);
        myfile << numberOfVertices << "\n";
        myfile << numberOfEdges << "\n";

        for(int i=0; i < numberOfVertices; i++) {
            for(int j=0; j < numberOfVertices; j++) {
                if(isDirectlyConnected(i, j)) {
                    myfile << i << "\t" << j << "\n";
                }
            }
        }

        myfile.close();
    }

    void Graph::writeGML(char* file_name) {
        ofstream myfile;
        myfile.open(file_name);
        myfile << "graph {" << "\n";

        for(int i=0; i < numberOfVertices; i++) {
            for(int j=0; j < numberOfVertices; j++) {
                if(isDirectlyConnected(i, j)) {
                    myfile << "\t" << i <<" -- " << j << ";\n";
                }
            }
        }

        myfile << "}" << "\n";
        myfile.close();
    }

    void Graph::setType(int dir, int lp) {
        direction = dir;
        loop = lp;
    }

    int Graph::getDir() {
        return direction;
    }

    int Graph::getLp() {
        return loop;
    }

    void Graph::clear(void) {
        delete content;
        numberOfEdges = 0;
        content = new int[size];
        fill(content, content+size, 0);
    }

    void Graph::addEdge(int v1, int v2) {
        if(isFullyConnected())
            throw new GraphEdgeOutOfBoundsException(size, pow(numberOfEdges, 2));

        checkVertixName(v1);
        checkVertixName(v2);
        content[v1 * numberOfVertices + v2] = 1;

        if (direction == UN_DIRECTED)
            content[v2 * numberOfVertices + v1] = 1;
        numberOfEdges++;
    }

    void Graph::removeEdge(int v1, int v2) {
        if(isEmpty())
            throw new GraphEdgeOutOfBoundsException(size, 0);

        checkVertixName(v1);
        checkVertixName(v2);

        if(isDirectlyConnected(v1, v2)) {
            content[v1 * numberOfVertices + v2] = 0;

            if (direction == UN_DIRECTED)
                content[v2 * numberOfVertices + v1] = 0;
            numberOfEdges--;
        }
    }

    void Graph::printGraphAsArray(void) {
        for(int i = 1; i <= size; i++) {
            cout << content[i-1] << " ";

            if (i % numberOfVertices == 0)
                cout << endl;
        }
    }

    int Graph::adjacentNodes(int v) {
        int ret = 0;
        for (int i=0; i<numberOfVertices; i++) {
            if(isDirectlyConnected(v, i))
                ret++;
        }
        return ret;
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

    bool Graph::isFullyConnected(void) {
        if(numberOfEdges == pow(numberOfVertices, 2))
            return true;

        return false;
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
        checkEdgeRange(E);
        srand(time(0));
        double theta;

        int v1, v2;
        for(int i = 0; i < E; i++) {
            theta = (double)rand() / RAND_MAX;

            if (theta < p) {
                v1 = i / numberOfVertices;
                v2 = i % numberOfVertices;
                addEdge(v1, v2);
            }
        }
    }

    void Graph::fillByZER(int E, double p) {
        checkEdgeRange(E);
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
                addEdge(v1, v2);
            }
        }
    }

    void Graph::fillByPreLogZER(int E, double p) {
        checkEdgeRange(E);
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
                addEdge(v1, v2);
            }
        }
    }

    void Graph::fillByPreZER(int E, double p, int m) {
        checkEdgeRange(E);
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
                addEdge(v1, v2);
            }
        }
    }

    int Graph::getSize(void) {
        return size;
    }

    int *Graph::getContent(void) {
        return content;
    }

    int Graph::getNumberOfEdges(void) {
        return numberOfEdges;
    }

    int Graph::getNumberOfVertices(void) {
        return numberOfVertices;
    }


    void Graph::checkVertixName(int vert) {
        if (vert < 0 || vert >= numberOfVertices)
            throw new GraphVertexOutOfBoundsException(numberOfVertices, vert);
    }

    void Graph::checkEdgeRange(int edge) {
        if (edge < 0 || edge > size)
            throw new GraphEdgeOutOfBoundsException(size, edge);
    }
}
