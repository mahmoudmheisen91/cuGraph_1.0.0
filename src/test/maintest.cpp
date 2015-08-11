/*
 * maintest.cpp
 *
 *  Created: 2015-05-24, Modified: 2015-08-11
 *
 */
 
// Standard C++ libraries includes:
#include <iostream>

// Headers includes:
#include <test/functional_test.h>

// Namespaces:
using namespace std;

// Main enrty:
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
