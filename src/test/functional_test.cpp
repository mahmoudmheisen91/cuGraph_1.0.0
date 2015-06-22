#include <iostream>
#include <test/functional_test.h>
#include <main/Graph.h>
#include <main/Exceptions.h>
#include <cuda/Parallel_functions.h>
#include <assert.h>

using namespace std;
using namespace cuGraph;

void test_empty_constructor(void) {
    Graph g1;

    try {
        g1.isEmpty();
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    try {
        g1.isFull();
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    assert(g1.getDirection() == UN_DIRECTED);
    assert(g1.getLoop() == SELF_LOOP);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.getNumberOfVertices() == 0);

    cout<<"End of: test_empty_constructor" << endl;
}

void test_empty_constructor_with_setget_methods(void) {
    Graph g1;
    g1.setType(UN_DIRECTED, SELF_LOOP);
    g1.setNumberOfVertices(10);

    try {
        g1.setNumberOfVertices(NULL);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }
    assert(g1.getDirection() == UN_DIRECTED);
    assert(g1.getLoop() == SELF_LOOP);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.getNumberOfVertices() == 10);

    cout<<"End of: test_empty_constructor_with_setget_methods" << endl;
}

void test_sec_constructor(void) {
    Graph g1(10);

    assert(g1.isEmpty());
    assert(!g1.isFull());
    assert(g1.getDirection() == UN_DIRECTED);
    assert(g1.getLoop() == SELF_LOOP);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.getNumberOfVertices() == 10);

    try {
        Graph g2(-10);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    try {
        Graph g3(0);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    try {
        Graph g4(NULL);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    cout<<"End of: test_sec_constructor" << endl;
}

void test_setType_method(void) {
    Graph g1;
    g1.setType(DIRECTED, NO_SELF_LOOP);

    assert(g1.getDirection() == DIRECTED);
    assert(g1.getLoop() == NO_SELF_LOOP);

    Graph g2;

    assert(g2.getDirection() == UN_DIRECTED);
    assert(g2.getLoop() == SELF_LOOP);

    Graph g3(10);
    g3.addEdge(1, 2);
    g3.setType(DIRECTED, SELF_LOOP);

    assert(g3.getDirection() == DIRECTED);
    assert(g3.getLoop() == SELF_LOOP);
    assert(g3.isEmpty());
    assert(!g3.isFull());

    g1.clear();
    assert(g1.getDirection() == DIRECTED);
    assert(g1.getLoop() == NO_SELF_LOOP);

    g2.clear();
    assert(g2.getDirection() == UN_DIRECTED);
    assert(g2.getLoop() == SELF_LOOP);

    g3.clear();
    assert(g3.getDirection() == DIRECTED);
    assert(g3.getLoop() == SELF_LOOP);
    assert(g3.isEmpty());
    assert(!g3.isFull());

    try {
        g1.setType(12, NO_SELF_LOOP);
    }
    catch(GraphDirectionTypeException* e) {
        cout<<"Caught the exception: GraphDirectionTypeException" << endl;
    }

    try {
        g2.setType(5645, 123);
    }
    catch(exception* e) {
        cout<<"Caught the exception: exception" << endl;
    }

    try {
        g3.setType(DIRECTED, 123);
    }
    catch(GraphLoopTypeException* e) {
        cout<<"Caught the exception: GraphLoopTypeException" << endl;
    }

    cout<<"End of: test_setType_method" << endl;
}

void test_setNumberOfVertices_method(void) {
    Graph g1;
    g1.setType(DIRECTED, NO_SELF_LOOP);
    g1.setNumberOfVertices(10);

    assert(g1.getNumberOfVertices() == 10);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.isEmpty());
    assert(!g1.isFull());

    Graph g2(10);
    assert(g2.getNumberOfVertices() == 10);
    assert(g2.getNumberOfEdges() == 0);
    assert(g2.isEmpty());
    assert(!g2.isFull());

    Graph g3;
    g3.setType(DIRECTED, NO_SELF_LOOP);
    try {
        g3.setNumberOfVertices(NULL);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    assert(g3.getNumberOfVertices() == 0);
    assert(g3.getNumberOfEdges() == 0);

    try {
        g3.isEmpty();
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    try {
        g3.isFull();
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    Graph g4;
    try {
        g4.setNumberOfVertices(NULL);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    try {
        g4.isEmpty();
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    try {
        g4.isFull();
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    assert(g4.getNumberOfVertices() == 0);
    assert(g4.getNumberOfEdges() == 0);

    try {
        Graph g5(NULL);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    Graph g6;
    g6.setType(DIRECTED, NO_SELF_LOOP);
    try {
        g6.setNumberOfVertices(-544);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    assert(g6.getNumberOfVertices() == 0);
    assert(g6.getNumberOfEdges() == 0);

    Graph g7;
    g7.setType(DIRECTED, NO_SELF_LOOP);
    try {
        g7.setNumberOfVertices(0);
    }
    catch(GraphNumberOfVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphNumberOfVertexOutOfBoundsException" << endl;
    }

    assert(g7.getNumberOfVertices() == 0);
    assert(g7.getNumberOfEdges() == 0);

    cout<<"End of: test_setNumberOfVertices_method" << endl;
}

void test_clear_method(void) {
    Graph g1;

    g1.clear();
    assert(g1.getNumberOfVertices() == 0);
    assert(g1.getNumberOfEdges() == 0);

    g1.setNumberOfVertices(10);
    assert(g1.getNumberOfVertices() == 10);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.isEmpty());
    assert(!g1.isFull());

    g1.clear();
    assert(g1.getNumberOfVertices() == 10);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.isEmpty());
    assert(!g1.isFull());

    g1.setNumberOfVertices(10);
    g1.addEdge(1, 3);
    assert(g1.getNumberOfVertices() == 10);
    assert(g1.getNumberOfEdges() == 1);
    assert(!g1.isEmpty());
    assert(!g1.isFull());

    g1.clear();
    assert(g1.getNumberOfVertices() == 10);
    assert(g1.getNumberOfEdges() == 0);
    assert(g1.isEmpty());
    assert(!g1.isFull());

    Graph g2(5);

    g2.clear();
    assert(g2.getNumberOfVertices() == 5);
    assert(g2.getNumberOfEdges() == 0);
    assert(g2.isEmpty());
    assert(!g2.isFull());

    cout<<"End of: test_clear_method" << endl;
}

void test_add_remove_methods(void) {
    Graph g1;

    try {
        g1.addEdge(1, 3);
    }
    catch(GraphIsNotInitException* e) {
        cout<<"Caught the exception: GraphIsNotInitException" << endl;
    }

    g1.setNumberOfVertices(4);
    g1.addEdge(0, 2);
    g1.addEdge(1, 2);

    assert(g1.getNumberOfEdges() == 2);

    g1.removeEdge(1, 2);
    assert(g1.getNumberOfEdges() == 1);
    g1.clear();
    assert(g1.getNumberOfEdges() == 0);

    try {
        g1.addEdge(-1, 3);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    try {
        g1.addEdge(0, 5);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    try {
        g1.addEdge(0, 4);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    try {
        g1.addEdge(5, 3);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    g1.addEdge(1, 2);

    try {
        g1.removeEdge(-1, 3);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    try {
        g1.removeEdge(0, 5);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    try {
        g1.removeEdge(0, 4);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    try {
        g1.removeEdge(5, 3);
    }
    catch(GraphVertexOutOfBoundsException* e) {
        cout<<"Caught the exception: GraphVertexOutOfBoundsException" << endl;
    }

    g1.addEdge(1, 2);
    g1.addEdge(1, 1);
    g1.setType(DIRECTED, NO_SELF_LOOP);
    assert(g1.isEmpty());

    try {
        g1.addEdge(1, 1);
    }
    catch(GraphLoopTypeException* e) {
        cout<<"Caught the exception: GraphLoopTypeException" << endl;
    }

    cout<<"End of: test_add_remove_methods" << endl;
}
























