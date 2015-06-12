#ifndef GRAPHSTREAM_H
#define GRAPHSTREAM_H

#include <iostream>
#include <string>
#include <fstream>
#include "Graph.h"

using namespace std;

namespace cuGraph {

    class GraphIO {
        public:
            GraphIO(string name);
//            virtual ~GraphIO();
            ofstream& operator<<(Graph &g);

        protected:

        private:
            string local_name;
    };
}

#endif // GRAPHSTREAM_H
