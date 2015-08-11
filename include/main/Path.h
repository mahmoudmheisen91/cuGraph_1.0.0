/*
 * Path.h
 *
 *  Created: 2015-05-24, Modified: 2015-08-11
 *
 */
#ifndef PATH_H_
#define PATH_H_

// Headers includes:
#include "Graph.h"

namespace cuGraph {

	// depth first search to find a path between two nodes:
    class Path {

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
}
#endif // PATH_H_
