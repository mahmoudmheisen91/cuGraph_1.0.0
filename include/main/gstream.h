#ifndef GSTREAM_H
#define GSTREAM_H

#include <string>
#include <fstream>
#include "Graph.h"

using namespace std;

namespace cuGraph {

    class ogstream {
        public:
            ogstream();
            ofstream& operator<<(Graph &g);
            void open(string name);
            void close(void);

        protected:

        private:
            string local_name;
            ofstream myfile;
            void toTXT(Graph *g);
            void toGML(Graph *g);
            void toMTX(Graph *g);
    };

    class igstream {
        public:
            igstream();
            ifstream& operator>>(Graph &g);
            void open(string name);
            void close(void);

        protected:

        private:
            string local_name;
            ifstream myfile;
            void fromTXT(Graph *g);
            void fromGML(Graph *g);
            void fromMTX(Graph *g);
    };
}

#endif // GSTREAM_H
