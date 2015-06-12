#include "Path.h"
#include <algorithm>
#include <iostream>

using namespace std;

namespace cuGraph {

    // Find paths in G from s:
    Path::Path(Graph *G, int s) {
        // init data structures:
        fromHere = s;
        size = G->getNumberOfVertices();
        visited = new bool[size];
        edgeTo = new int[size];

        fill(visited, visited+size, false);
        fill(edgeTo, edgeTo+size, -1);

        // find vertices connected to s:
        dfs(G, s);
    }

    Path::~Path() {
        delete visited;
        delete edgeTo;
    }

    void Path::dfs(Graph *G, int u) {
        visited[u] = true;
        int *content = G->getContent();
        for(int v = 0; v < size; v++) {
            if(!visited[v] && content[u * size + v]) {
                dfs(G, v);
                edgeTo[v] = u;
            }
        }
    }

    bool Path::hasPathTo(int v) {
        return visited[v];
    }

    // TODO: know the path
    /*
     public Iterable<Integer> pathTo(int v)
     {
     if (!hasPathTo(v)) return null;
     Stack<Integer> path = new Stack<Integer>();
     for (int x = v; x != s; x = edgeTo[x])
     path.push(x);
     path.push(s);
     return path;
     }
    */
}


