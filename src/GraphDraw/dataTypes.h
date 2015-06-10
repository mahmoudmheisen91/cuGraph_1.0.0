#ifndef DATATYPES_HPP
#define DATATYPES_HPP

namespace cuGraph {

    struct Color {
        int red;
        int green;
        int blue;

        Color() {
            red = 0; green = 0; blue = 0;
        }

        Color(int r, int g, int b) {
            red = r; green = g; blue = b;
        }

        void setColor(int r, int g, int b) {
            red = r; green = g; blue = b;
        }
    };

    struct Point {
        int x, y;

        Point() {
            x = 0; y = 0;
        }

        Point(int x0, int y0) {
            x = x0; y = y0;
        }
    };
}

#endif // DATATYPES_HPP
