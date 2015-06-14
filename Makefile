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
PATHC = _release/Path.o
EXCEP = _release/Exceptions.o
APP = output/bin/main/cuGraph_1.0.0

# Shell Commands:
all: $(EDITOR) $(EXCEP) $(GRAPH) $(PATHC) $(APP)
test: $(EDITOR) $(EXCEP) $(GRAPH) $(PATHC) $(TEST)

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
		
# Build Commands:	
$(EDITOR): src/main/Editor.cpp include/main/Editor.h
	$(PATCH)
	$(QT4C) include/main/Editor.h | \
	$(CPP) $(OFLAGS) $(INCFLAGS) -c -x c++ - -include src/main/Editor.cpp -o $(EDITOR)
	
$(EXCEP): src/main/Exceptions.cpp 
	$(CPP) $(INCFLAGS) -o $(EXCEP) -c src/main/Exceptions.cpp
	
$(GRAPH): src/main/Graph.cpp \
include/main/Path.h _release/Exceptions.o include/main/gstream.h
	$(CPP) $(INCFLAGS) -o $(GRAPH) -c src/main/Graph.cpp
	
$(PATHC): src/main/Path.cpp _release/Graph.o
	$(CPP) $(INCFLAGS) -o $(PATHC) -c src/main/Path.cpp

$(APP): main.cpp \
_release/Graph.o _release/Path.o _release/Exceptions.o
	$(CPP) $^ $(INCFLAGS) -o $(APP)
	
#$(EXECUTABLE): $(CPPSOURCES) src/Editor.h
#	$(CPP) $^ $(INCL) src/Editor.o $(FLAGS) -o bin/$@

