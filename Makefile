# Compilers Definitions:
CPP  = g++
QT4C = moc-qt4
NVCC = nvcc

# Compilers Flags:
FLAGS = -g -Wall -lQtCore -lQtGui
INCFLAGS = -I./include

# Library Directories:
INCL = -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui
OFLAGS = $(INCL) -Wall -Wno-unreachable-code -Wno-return-type

EDITOR = _release/Editor.o
GRAPH  = _release/Graph.o
GRAPHDRAW  = _release/GraphDraw.o
PATHC = _release/Path.o
EXCEP = _release/Exceptions.o
APP = output/bin/main/cuGraph_1.0.0
FUNC_TEST = output/bin/test/functional_test
SCAN = _release/Scan.o

# Shell Commands:
all: $(SCAN) $(EXCEP) $(EDITOR) $(GRAPH) $(GRAPHDRAW) $(PATHC) $(APP)
functional_test: $(EXCEP) $(EDITOR) $(GRAPH) $(GRAPHDRAW) $(PATHC) $(FUNC_TEST)

clean:
	$(RM) $(shell pwd)/output/GML/*~
	$(RM) $(shell pwd)/output/TXT/*~
	$(RM) $(shell pwd)/output/bin/main/*
	$(RM) $(shell pwd)/output/bin/test/*
	$(RM) $(shell pwd)/_debug/*
	$(RM) $(shell pwd)/_release/*
	$(RM) $(shell pwd)/src/*~ 
	$(RM) $(shell pwd)/src/cuda/*~ 
	$(RM) $(shell pwd)/src/main/*~ 
	$(RM) $(shell pwd)/src/test/*~ 
	$(RM) $(shell pwd)/include/*~ 
	$(RM) $(shell pwd)/include/main/*~ 
	$(RM) $(shell pwd)/include/test/*~ 
	$(RM) $(shell pwd)/*~ 
	$(RM) $(shell pwd)/*.png 

run: 
	./$(APP)
	
run_functional_test: 
	./$(FUNC_TEST)
		
# Build Commands:	
$(SCAN): src/cuda/scan.cu
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/Scan.cu
	
$(EXCEP): src/main/Exceptions.cpp 
	$(CPP) $(INCFLAGS) -o $(EXCEP) -c src/main/Exceptions.cpp
	
$(EDITOR): src/main/Editor.cpp include/main/Editor.h
	$(PATCH)
	$(QT4C) include/main/Editor.h | \
	$(CPP) $(OFLAGS) $(INCFLAGS) -c -x c++ - -include src/main/Editor.cpp -o $(EDITOR)
	
$(GRAPH): src/main/Graph.cpp \
include/main/Path.h _release/Exceptions.o include/main/gstream.h
	$(CPP) $(INCFLAGS) -o $(GRAPH) -c src/main/Graph.cpp -DDEBUG

$(GRAPHDRAW): src/main/GraphDraw.cpp _release/Graph.o \
_release/Editor.o include/main/Editor.h
	$(CPP) $(INCFLAGS) $(INCL) $(FLAGS) -o $(GRAPHDRAW) -c src/main/GraphDraw.cpp
	
$(PATHC): src/main/Path.cpp _release/Graph.o
	$(CPP) $(INCFLAGS) -o $(PATHC) -c src/main/Path.cpp

$(APP): main.cpp \
_release/Graph.o _release/GraphDraw.o _release/Editor.o _release/Path.o _release/Exceptions.o
	$(CPP) $^ $(INCFLAGS) $(OFLAGS) -o $(APP) $(FLAGS) -DDEBUG
	
$(FUNC_TEST): src/test/maintest.cpp src/test/functional_test.cpp \
_release/Graph.o _release/Path.o _release/Exceptions.o include/test/functional_test.h
	$(CPP) $^ $(INCFLAGS) -o $(FUNC_TEST)
	
#$(EXECUTABLE): $(CPPSOURCES) src/Editor.h
#	$(CPP) $^ $(INCL) src/Editor.o $(FLAGS) -o bin/$@

