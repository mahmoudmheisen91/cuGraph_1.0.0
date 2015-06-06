# Compilers Definitions:
CPP  = g++
QT4C = moc-qt4
NVCC = nvcc

# Compilers Flags:
CFLAGS1   = -c -Wall 
CFLAGS2   = -c -x c++ - -include
LIBSFLAGS = -lm -lQtCore -lQtGui 

# Library Directories:
HEADERS = -I$(shell pwd)/inc 
STDLIBS = -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui 
PRJLIBS = -I$(shell pwd)/lib 

# Project Directories:
DRAWCPP := $(shell pwd)/lib/draw.cpp 
DRAWH   := $(shell pwd)/lib/draw.h
DRAWO   := $(shell pwd)/obj/draw.o

# C++ Paths:
vpath %.cpp $(shell pwd)/src

# C++ Sources:
CPPSOURCES += main.cpp GraphArray.cpp Path.cpp GraphVertexOutOfBoundsException.cpp
CPPSOURCES += GraphEdgeOutOfBoundsException.cpp 

# C++ Objects:
OBJECTS = $(patsubst %.cpp,obj/%.o,$(CPPSOURCES))

# Executable file:
EXECUTABLE = cuGraph_1.0.0

# Shell Commands:
all: obj/draw.o $(EXECUTABLE) examples/polygon

run: 
	./bin/$(EXECUTABLE)

clean:
	$(RM) $(shell pwd)/obj/*.o 
	$(RM) $(shell pwd)/bin/*
	$(RM) $(shell pwd)/src/*~ 
	$(RM) $(shell pwd)/src/cuda/*~ 

# Build Commands:
obj/draw.o: $(DRAWCPP) $(DRAWH) 
	$(QT4C) $(DRAWCPP) | $(CPP) $(STDLIBS) $(CFLAGS2) $(DRAWCPP) -o $@

obj/%.o : %.cpp
	$(CPP) -o $@ $< $(CFLAGS1) $(HEADERS)	
	
$(EXECUTABLE): $(OBJECTS)
	$(OBJECTS) | $(CPP) -o bin/$@ $^ $(HEADERS) $(LIBSFLAGS)

examples/polygon: $(DRAWO)
	$(CPP) -o bin/polygon $^ $(DRAWH) $(LIBSFLAGS) $(PRJLIBS) examples/polygon.cpp



