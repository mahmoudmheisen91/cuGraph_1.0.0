#include "Path.h"
#include <algorithm>

// Find paths in G from s:
Path::Path(GraphArray G, int s) {
	// init data structures:
	this.s = s;
	size = G.getNumberOfVertices();
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

