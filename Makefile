# define the C++ compiler to use:
CC = g++

# define any compile-time flags:
CFLAGS = -c -Wall

# define any directories containing header files other than /usr/include:
INCLUDES = -I$(shell pwd)/inc -I$(shell pwd)/data

# define library paths in addition to /usr/lib:
LFLAGS =

# define any libraries to link into executable:
LIBS = -lm

# define paths for .c files:
vpath %.cpp $(shell pwd)/src

# define the C source files:
SOURCES = main.cpp GraphArray.cpp Path.cpp GraphVertexOutOfBoundsException.cpp \
			GraphEdgeOutOfBoundsException.cpp

# define the C object files:
OBJECTS = $(patsubst %.cpp,obj/%.o,$(SOURCES))

# define the executable file:
EXECUTABLE = cuGraph_1.0.0

GRAPH: $(SOURCES) $(EXECUTABLE)

# build executable:
$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $^ $(INCLUDES) $(LIBS) $(LFLAGS)

# if any OBJECTS must be built then obj must be built first:
$(OBJECTS): | obj

# replacement rule for building .o's from .cpp's:
obj/%.o : %.cpp
	$(CC) -o $@ $< $(CFLAGS) $(INCLUDES)

# run command:
run:
	./$(EXECUTABLE)

# clean command: remove object files:
clean:
	$(RM) $(shell pwd)/obj/*.o 
	$(RM) $(shell pwd)/src/*~ 
	$(RM) *~ $(EXECUTABLE)

