#include "gstream.h"

namespace cuGraph {

    ogstream::ogstream() {
    }

    ofstream &ogstream::operator <<(Graph &g) {
        string str1 = local_name.substr(local_name.length() - 4,local_name.length() - 4);
        string str2 = ".txt";
        string str3 = ".gml";

        if(str1.compare(str2) == 0)
            toTXT(&g);
        else if(str1.compare(str3) == 0)
            toGML(&g);
        else
            toTXT(&g);

        return myfile;
    }

    void ogstream::open(string name) {
        local_name = name;
        char *cstr = &local_name[0u];
        myfile.open(cstr);
    }

    void ogstream::close(void) {
        myfile.close();
    }

    void ogstream::toTXT(Graph *g) {
        myfile << g->numberOfVertices << "\n";
        myfile << g->numberOfEdges << "\n";

        for(int i=0; i < g->numberOfVertices; i++) {
            for(int j=0; j < g->numberOfVertices; j++) {
                if(g->isDirectlyConnected(i, j)) {
                    myfile << i << "\t" << j << "\n";
                }
            }
        }
    }

    void ogstream::toGML(Graph *g) {

        myfile << "graph {" << "\n";

        if(g->direction == UN_DIRECTED) {
            for(int i=0; i < g->numberOfVertices; i++) {
                for(int j=0; j < g->numberOfVertices; j++) {
                    if(g->isDirectlyConnected(i, j)) {
                        myfile << "\t" << i <<" -- " << j << ";\n";
                    }
                }
            }
        }
        else {
            for(int i=0; i < g->numberOfVertices; i++) {
                for(int j=0; j < g->numberOfVertices; j++) {
                    if(g->isDirectlyConnected(i, j)) {
                        myfile << "\t" << i <<" -> " << j << ";\n";
                    }
                }
            }
        }

        myfile << "}" << "\n";
    }

    igstream::igstream() {

    }

    ifstream &igstream::operator >>(Graph &g) {
        string str1 = local_name.substr(local_name.length() - 4,local_name.length() - 4);
        string str2 = ".txt";
        string str3 = ".gml";

        if(str1.compare(str2) == 0)
            fromTXT(&g);
        else if(str1.compare(str3) == 0)
            fromGML(&g);
        else
            fromTXT(&g);

        return myfile;
    }

    void igstream::open(string name) {
        local_name = name;
        char *cstr = &local_name[0u];
        myfile.open(cstr);
    }

    void igstream::close(void) {
        myfile.close();
    }

    void igstream::fromTXT(Graph *g) {

        myfile >> g->numberOfVertices;
        myfile >> g->numberOfEdges;

        int v1, v2;
        while (myfile >> v1) {
            myfile >> v2;
            g->content[v1 * g->numberOfVertices + v2] = 1;
        }
        myfile.close();
    }

    void igstream::fromGML(Graph *g) {

    }
}
