#include <iostream>
#include "test_graph.h"
#include "Graph.h"

using namespace std;
using namespace cuGraph;

int main(int argc, char** argv) {

    test_empty_constructor();
    test_empty_constructor_with_setget_methods();
    test_sec_constructor();

    test_setType_method();
    test_setNumberOfVertices_method();
    test_clear_method();
    test_add_remove_methods();

    return 0;
}
