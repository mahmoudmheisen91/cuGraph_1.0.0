#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "Path.h"
#include "Graph.h"
#include "Exceptions.h"

using namespace std;
using namespace draw;
Graph::Graph() {}

Graph::Graph(int V) :numberOfVertices(V) {
	size = pow(numberOfVertices, 2);
	numberOfEdges = 0;
	content = new int[size];

	fill(content, content+size, 0);

    // Default values:
    direction = UN_DIRECTED;
    loop = SELF_LOOP;
    defaultSettings.rangeMin = -1000;
    defaultSettings.rangeMax = 1000;
    defaultSettings.color = BLACK;
    defaultSettings.penWidth = 5;
    defaultSettings.transparency = 0;
    defaultSettings.fontSize = 12;
    defaultSettings.windowWidth = 512;
    defaultSettings.windowHeight = 512;
}

Graph::~Graph(void) {
	delete content;
}

void Graph::setType(int dir, int lp) {
    direction = dir;
    loop = lp;
}

void Graph::setDrawSettings(Settings sets) {
    defaultSettings.rangeMin = sets.rangeMin;
    defaultSettings.rangeMax = sets.rangeMax;
    defaultSettings.color = sets.color;
    defaultSettings.penWidth = sets.penWidth;
    defaultSettings.transparency = sets.transparency;
    defaultSettings.fontSize = sets.fontSize;
    defaultSettings.windowWidth = sets.windowWidth;
    defaultSettings.windowHeight = sets.windowHeight;
}

void Graph::draw(void) {
    setrange(defaultSettings.rangeMin, defaultSettings.rangeMax);
    setcolor(defaultSettings.color);
    setpenwidth(defaultSettings.penWidth);
    settransparency(defaultSettings.transparency);
    setfontsize(defaultSettings.fontSize);
    setwindowsize(defaultSettings.windowWidth, defaultSettings.windowHeight);

    // Pseudo-code for the algorithm by Fruchterman and Reingold [FR91]:
    // http://cs.brown.edu/~rt/gdhandbook/chapters/force-directed.pdf - page 5:
    int W = defaultSettings.windowWidth;
    int L = defaultSettings.windowHeight;
    int area = W * L;
    double **g = (double **)malloc(numberOfVertices * sizeof(double *));
    for (int i=0; i<numberOfVertices; i++)
        g[i] = (double *)malloc(4 * sizeof(double));

    // g[0] = disp.x, g[1] = disp.y, g[2] = pos.x, g[3] = pos.y
    for (int i=0; i<numberOfVertices; i++) {
        g[i][2] = (rand() / (double)RAND_MAX) * W;
        g[i][3] = (rand() / (double)RAND_MAX) * L;
        cout << "rnd" << g[i][2] << " " << g[i][3] << endl;
    }

    double K = sqrt(area/numberOfVertices);
    int iterations = 1;
    int t = 1000;
    double segma[2];

    for(int i = 1; i <= iterations; i++) {
        // calculate repulsive forces:
        for (int v=0; v<numberOfVertices; v++) {
            g[v][0] = 0;
            g[v][1] = 0;
            for (int u=0; u<numberOfVertices; u++) {
                if (u != v) {
                    segma[0] = g[v][2] - g[u][2];
                    segma[1] = g[v][3] - g[u][3];

                    g[v][0] += (segma[0] / length(segma)) * fr(length(segma), K);
                    g[v][1] += (segma[1] / length(segma)) * fr(length(segma), K);
                }
            }
        }

        // calculate attractive forces:
        for (int v=0; v<numberOfVertices; v++) {
            for (int u=0; u<numberOfVertices; u++) {
                if (isDirectlyConnected(v, u)) {
                    segma[0] = g[v][2] - g[u][2];
                    segma[1] = g[v][3] - g[u][3];

                    g[v][0] -= (segma[0] / length(segma)) * fa(length(segma), K);
                    g[v][1] -= (segma[1] / length(segma)) * fa(length(segma), K);

                    g[u][0] += (segma[0] / length(segma)) * fa(length(segma), K);
                    g[u][1] += (segma[1] / length(segma)) * fa(length(segma), K);
                }
            }
        }

        // limit max displacement to temperature t:
        for (int v=0; v<numberOfVertices; v++) {
            double array[2] = {g[v][0], g[v][1]};
            g[v][2] += (g[v][0] / length(array)) * min(g[v][0], (double)t);
            g[v][3] += (g[v][1] / length(array)) * min(g[v][1], (double)t);

            g[v][2] = min((double)W/2, max(-W/(double)2, g[v][2]));
            g[v][3] = min((double)L/2, max(-L/(double)2, g[v][3]));
        }

        // reduce the temperature as the layout approaches a better configuration:
        t = cool(t);
    }

    for (int v=0; v<numberOfVertices; v++) {
        cout << g[v][2] << " " << g[v][3] << endl;
        point(g[v][2], g[v][3]);
    }

        /*
        area:= W ∗ L; {W and L are the width and length of the frame}
        G := (V, E); {the vertices are assigned random initial positions}
        k := p area/|V |;
        function fa(x) := begin return x2/k end;
        function fr(x) := begin return k2/x end;
        for i := 1 to iterations do begin
        {calculate repulsive forces}
        for v in V do begin
        {each vertex has two vectors: .pos and .disp
        v.disp := 0;
        for u in V do
        if (u 6= v) then begin
        {δ is the difference vector between the positions of the two vertices}
        δ := v.pos − u.pos;
        v.disp := v.disp + (δ/|δ|) ∗ fr(|δ|)
        end
        end
        {calculate attractive forces}
        for e in E do begin
        {each edges is an ordered pair of vertices .vand.u}
        δ := e.v.pos − e.u.pos;
        e.v.disp := e.v.disp − (δ/|δ|) ∗ fa(|δ|);
        e.u.disp := e.u.disp + (δ/|δ|) ∗ fa(|δ|)
        end
        {limit max displacement to temperature t and prevent from displacement outside frame}
        for v in V do begin
        v.pos := v.pos + (v.disp/|v.disp|) ∗ min(v.disp, t);
        v.pos.x := min(W/2, max(−W/2, v.pos.x));
        v.pos.y := min(L/2, max(−L/2, v.pos.y))
        end
        {reduce the temperature as the layout approaches a better configuration}
        t := cool(t)
        end
        */
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
	double *logp, c;

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

double fa(double x, double k) {
    return pow(x, 2) / k;
}

double fr(double x, double k) {
    return pow(k, 2) / x ;
}

double length(double array[2]) {
    return sqrt(pow(array[0], 2) + pow(array[1], 2));
}

int cool(int t) {
    return t;
}



