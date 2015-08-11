/*
 * gstream.cpp
 *
 *  Created: 2015-05-24, Modified: 2015-08-10
 *
 */
 
// Headers includes:
#include <main/gstream.h>
#include <main/dataTypes.h>

namespace cuGraph {

	// Output Graph stream:
    ogstream::ogstream() {
    }

	// overload <<:
    ofstream &ogstream::operator <<(Graph &g) {
        string str1 = local_name.substr(local_name.length() - 4,local_name.length() - 4);

        if(!str1.compare(".txt"))
            toTXT(&g);
        else if(!str1.compare(".gml"))
            toGML(&g);
        else if(!str1.compare(".mtx"))
            toMTX(&g);
        else
            toTXT(&g);

        return myfile;
    }

	// open/create file:
    void ogstream::open(string name) {
        local_name = name;
        char *cstr = &local_name[0u];
        myfile.open(cstr);
    }

	// close:
    void ogstream::close(void) {
        myfile.close();
    }

	// to txt format:
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

	// to gml format:
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

	// to mtx format:
    void ogstream::toMTX(Graph *g) {
        myfile << g->numberOfVertices << " " << g->numberOfVertices << " ";
        myfile << g->countEdges() << "\n";

        for(int i=0; i < g->numberOfVertices; i++) {
            for(int j=0; j < g->numberOfVertices; j++) {
                if(g->isDirectlyConnected(i, j)) {
                    myfile << i+1 << "\t" << j+1 << "\n";
                }
            }
        }
    }

	// Input Graph stream:
    igstream::igstream() {

    }

	// overload >>:
    ifstream &igstream::operator >>(Graph &g) {
        string str1 = local_name.substr(local_name.length() - 4,local_name.length() - 4);

        if(!str1.compare(".txt"))
            fromTXT(&g);
        else if(!str1.compare(".gml"))
            fromGML(&g);
        else if(!str1.compare(".mtx"))
            fromMTX(&g);
        else
            fromTXT(&g);

        return myfile;
    }

	// open file:
    void igstream::open(string name) {
        local_name = name;
        char *cstr = &local_name[0u];
        myfile.open(cstr);
    }

	// close file:
    void igstream::close(void) {
        myfile.close();
    }

	// read from txt format:
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

	// read from gml format:
    void igstream::fromGML(Graph *g) {

    }

	// read from mtx format:
    void igstream::fromMTX(Graph *g) {

        int verts, edges;
        myfile >> verts;
        myfile >> verts;
        g->setNumberOfVertices(verts);

        myfile >> edges;

        int v1, v2;
        while (myfile >> v1) {
            myfile >> v2;
            g->addEdge(v1-1, v2-1);
        }

        g->setNumberOfEdges(edges);
        myfile.close();
    }
}
