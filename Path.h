#ifndef PATH_H_
#define PATH_H_

#include "GraphArray.h"

class Path
{
	public:
		Path(GraphArray *G, int s);
		virtual ~Path();
		bool hasPathTo(int v);
	protected:
	private:
		int fromHere;
		int size;
		bool* visited;
		int* edgeTo;
		void dfs(GraphArray *G, int v);
};

#endif // PATH_H_
