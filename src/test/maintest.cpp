#include <iostream>
#include <test/functional_test.h>

using namespace std;

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
