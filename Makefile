# Compilers Definitions:
CPP  = g++
QT4C = moc-qt4
NVCC = nvcc

# Compilers Flags:
CFLAGS    = -c -Wall 
LIBSFLAGS = -lm -lQtCore -lQtGui 

# Library Directories:
HEADERS = -I$(shell pwd)/inc 
STDLIBS = -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui 
PRJLIBS = -I$(shell pwd)/lib 

# Project Directories:
DRAWCPP := $(shell pwd)/lib/draw.cpp 
DRAWH   := $(shell pwd)/lib/draw.h
DRAWO   := $(shell pwd)/lib/draw.o

# C++ Paths:
vpath %.cpp $(shell pwd)/src

# C++ Sources:
CPPSOURCES += main.cpp Graph.cpp Path.cpp Exceptions.cpp

# Executable file:
EXECUTABLE = cuGraph_1.0.0

# Shell Commands:
all: lib/draw.o $(EXECUTABLE) 

run: 
	./bin/$(EXECUTABLE)

clean:
	$(RM) $(shell pwd)/lib/*.o 
	$(RM) $(shell pwd)/bin/*
	$(RM) $(shell pwd)/src/*~ 
	$(RM) $(shell pwd)/src/cuda/*~ 
	$(RM) $(shell pwd)/*.png 

# Build Commands:
lib/draw.o: $(DRAWCPP) $(DRAWH) 
	$(QT4C) $(DRAWCPP) | $(CPP) $(STDLIBS) $(CFLAGS) -x c++ - -include $(DRAWCPP) -o $@

$(EXECUTABLE): $(CPPSOURCES) $(DRAWO)
	$(CPP) -o bin/$@ $^ $(HEADERS) $(DRAWH) $(LIBSFLAGS) $(PRJLIBS)
	


