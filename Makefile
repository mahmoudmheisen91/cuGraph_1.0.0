# Compilers Definitions:
CPP  = g++
QT4C = moc-qt4
NVCC = nvcc

# Compilers Flags:
FLAGS = -g -Wall -lQtCore -lQtGui

# Library Directories:
INCL = -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui
OFLAGS = $(INCL) -Wall -Wno-unreachable-code -Wno-return-type

# C++ Sources:
CPPSOURCES += src/main.cpp src/Graph.cpp src/Path.cpp src/Exceptions.cpp src/GraphDraw.cpp src/graphStream.cpp

# Executable file:
EXECUTABLE = cuGraph_1.0.0

# Shell Commands:
all: src/Editor.o $(EXECUTABLE)

run: 
	./bin/$(EXECUTABLE)

clean:
	$(RM) $(shell pwd)/bin/*
	$(RM) $(shell pwd)/src/*~ 
	$(RM) $(shell pwd)/src/*.o 
	$(RM) $(shell pwd)/*~ 
	$(RM) $(shell pwd)/src/cuda/*~ 
	$(RM) $(shell pwd)/src/GraphDraw/*~ 
	$(RM) $(shell pwd)/*.png 

# Build Commands:
src/Editor.o: src/Editor.cpp src/Editor.h
	$(PATCH)
	$(QT4C) src/Editor.h | $(CXX) $(OFLAGS) -c -x c++ - -include src/Editor.cpp -o src/Editor.o
	
$(EXECUTABLE): $(CPPSOURCES) src/Editor.h
	$(CPP) $^ $(INCL) src/Editor.o $(FLAGS) -o bin/$@



