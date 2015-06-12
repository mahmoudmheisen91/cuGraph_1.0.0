#include "graphStream.h"

namespace cuGraph {

    GraphIO::GraphIO(string name) {
        local_name = name;
    }

//    GraphIO::~GraphIO() {
//        delete local_g;
//        delete local_file_name;
//    }

    ofstream &GraphIO::operator <<(Graph &g) {
        char *cstr = &local_name[0u];

        ofstream myfile;
        myfile.open(cstr);

        myfile << g.numberOfVertices << "\n";
        myfile << g.numberOfEdges << "\n";

        for(int i=0; i < g.numberOfVertices; i++) {
            for(int j=0; j < g.numberOfVertices; j++) {
                if(g.isDirectlyConnected(i, j)) {
                    myfile << i << "\t" << j << "\n";
                }
            }
        }

        myfile.close();
        return myfile;
    }
}
