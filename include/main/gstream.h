/*
 * gstream.h
 *
 *  Created: 2015-05-24, Modified: 2015-08-10
 *
 */
#ifndef GSTREAM_H
#define GSTREAM_H

// Standard C++ libraries includes:
#include <string>
#include <fstream>

// Headers includes:
#include "Graph.h"

// Namespaces:
using namespace std;

namespace cuGraph {

	// Output Graph stream:
    class ogstream {
        public:
            ogstream();							// constructor
            ofstream& operator<<(Graph &g);		// overload <<
            void open(string name);				// create/open file
            void close(void);					// close file

        protected:

        private:
            string local_name;
            ofstream myfile;
            void toTXT(Graph *g);				// txt format
            void toGML(Graph *g);				// gml format
            void toMTX(Graph *g);				// mtx format
    };

	// Input Graph stream:
    class igstream {
        public:
            igstream();							// constructor
            ifstream& operator>>(Graph &g);		// overload >>
            void open(string name);				// open file
            void close(void);					// close file

        protected:

        private:
            string local_name;
            ifstream myfile;
            void fromTXT(Graph *g);				// txt format
            void fromGML(Graph *g);				// gml format
            void fromMTX(Graph *g);				// mtx format
    };
}

#endif // GSTREAM_H
