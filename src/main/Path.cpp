#include <main/Path.h>
#include <algorithm>

namespace cuGraph {

    // Find paths in G from s:
    Path::Path(Graph *G, int s) {

        fromHere = s;
        size = G->numberOfVertices;
        visited = new bool[size];
        edgeTo = new int[size];

        std::fill(visited, visited+size, false);
        std::fill(edgeTo, edgeTo+size, -1);

        // find vertices connected to s:
        dfs(G, s);
    }

    Path::~Path() {
        delete visited;
        delete edgeTo;
    }

    void Path::dfs(Graph *G, int u) {
        visited[u] = true;
        bool *content = G->content;
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
}


