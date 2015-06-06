# define the C++ compiler to use:
CXX = g++
MOCQT4 = moc-qt4
FILES1 := $(shell pwd)/lib/draw.cpp 
FILES2 := $(shell pwd)/lib/draw.h
FILES3 := $(shell pwd)/obj/draw.o

# define any compile-time flags:
CFLAGS = -c -Wall

# define any directories containing header files other than /usr/include:
INCLUDES = -I$(shell pwd)/inc -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui

# define library paths in addition to /usr/lib:
LFLAGS = -I$(shell pwd)/lib

# define any libraries to link into executable:
LIBS = -lm -lQtCore -lQtGui

# define paths for .cpp files:
vpath %.cpp $(shell pwd)/src

# define the C++ source files:
SOURCES = main.cpp GraphArray.cpp Path.cpp GraphVertexOutOfBoundsException.cpp \
			GraphEdgeOutOfBoundsException.cpp 

# define the C++ object files:
OBJECTS = $(patsubst %.cpp,obj/%.o,$(SOURCES))

# define the executable file:
EXECUTABLE = cuGraph_1.0.0

GRAPH: $(SOURCES) $(EXECUTABLE)

# build executable:
$(EXECUTABLE): $(OBJECTS)
	$(CXX) -o bin/$@ $^ $(INCLUDES) $(LIBS) $(LFLAGS)

examples/polygon: $(FILES3)
	$(CXX) -o bin/polygon $^ $(FILES2) $(LIBS) $(LFLAGS) examples/polygon.cpp
	
# if any OBJECTS must be built then obj must be built first:
$(OBJECTS): | obj

# replacement rule for building .o's from .cpp's:
obj/%.o : %.cpp
	$(CXX) -o $@ $< $(CFLAGS) $(INCLUDES)

obj/draw.o: $(FILES1) $(FILES2) 
	$(MOCQT4) $(FILES1) | $(CXX) $(INCLUDES) -c -x c++ - -include $(FILES1) -o $@
	
# run command:
run:
	./bin/polygon

# clean command: remove object files:
clean:
	$(RM) $(shell pwd)/obj/*.o 
	$(RM) $(shell pwd)/src/*~ 
	$(RM) *~ $(EXECUTABLE)

