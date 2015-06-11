#ifndef PATH_H_
#define PATH_H_

#include "Graph.h"

class Path
{
	public:
        Path(Graph *G, int s);
		virtual ~Path();
		bool hasPathTo(int v);
	protected:
	private:
		int fromHere;
		int size;
		bool* visited;
		int* edgeTo;
        void dfs(Graph *G, int v);
};

#endif // PATH_H_
