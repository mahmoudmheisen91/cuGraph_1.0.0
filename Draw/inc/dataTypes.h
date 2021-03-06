#ifndef DATATYPES_HPP
#define DATATYPES_HPP

#define DIRECTED        0
#define UN_DIRECTED     1

namespace cuGraph {

    struct Color {
        int red, blue, green;

        Color() {}

        Color(int r, int g, int b) {
            red = r; green = g; blue = b;
        }

        void set(int r, int g, int b) {
            red = r; green = g; blue = b;
        }

        void set(Color c) {
            set(c.red, c.green, c.blue);
        }
    };

    struct Point {
        double x, y;

        Point() {}

        Point(double x0, double y0) {
            x = x0; y = y0;
        }

        void set(double x0, double y0) {
            x = x0; y = y0;
        }

        void set(Point p) {
            set(p.x, p.y);
        }
    };

    typedef struct _settings {
        Color node_color;
        Color edge_color;
        Color text_color;
        int line_width;
        int node_size;
        bool is_numbered;
    } Settings;
}

#endif // DATATYPES_HPP
