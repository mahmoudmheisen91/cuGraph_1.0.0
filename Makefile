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
SCANK = _release/scan_kernel.o
SCAN = _release/parallel_scan.o
RANDK = _release/random_number_generator_kernal.o
RAND = _release/parallel_generateRandomNumber.o
SKIPK = _release/skipValue_kernal.o
SKIP = _release/parallel_generateSkipValue.o

# Shell Commands:
cuda: $(SCANK) $(SCAN) $(RANDK) $(RAND) $(SKIPK) $(SKIP) 
all: $(SCANK) $(SCAN) $(RANDK) $(RAND) $(SKIPK) $(SKIP) $(EXCEP) $(EDITOR) $(GRAPH) $(GRAPHDRAW) $(PATHC) $(APP)
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
	$(RM) $(shell pwd)/include/cuda/*~ 
	$(RM) $(shell pwd)/*~ 
	$(RM) $(shell pwd)/*.png 

run: 
	./$(APP)
	
run_functional_test: 
	./$(FUNC_TEST)
		
# Build Commands:	
$(SCANK): src/cuda/scan_kernel.cu
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/scan_kernel.cu
	
$(SCAN): src/cuda/parallel_scan.cu _release/scan_kernel.o
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/parallel_scan.cu
	
$(RANDK): src/cuda/random_number_generator_kernal.cu
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/random_number_generator_kernal.cu
	
$(RAND): src/cuda/parallel_generateRandomNumber.cu _release/random_number_generator_kernal.o
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/parallel_generateRandomNumber.cu
	
$(SKIPK): src/cuda/skipValue_kernal.cu
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/skipValue_kernal.cu
	
$(SKIP): src/cuda/parallel_generateSkipValue.cu _release/skipValue_kernal.o
	$(NVCC) $(INCFLAGS) -o $@ -arch compute_20 -code sm_20 -c src/cuda/parallel_generateSkipValue.cu
	
$(EXCEP): src/main/Exceptions.cpp 
	$(CPP) $(INCFLAGS) -o $(EXCEP) -c src/main/Exceptions.cpp
	
$(EDITOR): src/main/Editor.cpp include/main/Editor.h
	$(PATCH)
	$(QT4C) include/main/Editor.h | \
	$(CPP) $(OFLAGS) $(INCFLAGS) -c -x c++ - -include src/main/Editor.cpp -o $(EDITOR)
	
$(GRAPH): src/main/Graph.cpp\
_release/random_number_generator_kernal.o _release/parallel_generateRandomNumber.o\
_release/skipValue_kernal.o _release/parallel_generateSkipValue.o \
include/main/Path.h _release/Exceptions.o include/main/gstream.h
	$(CPP) $(INCFLAGS) -o $(GRAPH) -L/usr/local/cuda/lib64 -lcuda -lcudart -c src/main/Graph.cpp -DDEBUG

$(GRAPHDRAW): src/main/GraphDraw.cpp _release/Graph.o \
_release/Editor.o include/main/Editor.h
	$(CPP) $(INCFLAGS) $(INCL) $(FLAGS) -o $(GRAPHDRAW) -c src/main/GraphDraw.cpp
	
$(PATHC): src/main/Path.cpp _release/Graph.o
	$(CPP) $(INCFLAGS) -o $(PATHC) -c src/main/Path.cpp

$(APP): main.cpp \
_release/random_number_generator_kernal.o _release/parallel_generateRandomNumber.o\
_release/skipValue_kernal.o _release/parallel_generateSkipValue.o \
_release/Graph.o _release/GraphDraw.o _release/Editor.o _release/Path.o _release/Exceptions.o
	$(CPP) $^ $(INCFLAGS) $(OFLAGS) -L/usr/local/cuda/lib64 -lcuda -lcudart -o $(APP) $(FLAGS) -DDEBUG
	
$(FUNC_TEST): src/test/maintest.cpp src/test/functional_test.cpp \
_release/Graph.o _release/Path.o _release/Exceptions.o include/test/functional_test.h
	$(CPP) $^ $(INCFLAGS) -o $(FUNC_TEST)
	
