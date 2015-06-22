# Compilers Definitions:
CPP  		= g++
QT4C 		= moc-qt4
NVCC 		= nvcc

# Compilers Flags:
CPPFLAGS 	= -g -Wall -fopenmp -DDEBUG
QTLIBS 		= -lQtCore -lQtGui
CUDAFLAGS 	= -arch compute_20 -code sm_20
CUDALIBS 	= -L/usr/local/cuda/lib64 -lcuda -lcudart
INCFLAGS 	= -I./include

# Libraries:
INCLQT 		= -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui
OFLAGS 		= $(INCLQT) -Wall -Wno-unreachable-code -Wno-return-type

# Directories:
SCANK 		= _release/scan_kernel.o 
RANDK 		= _release/random_number_generator_kernal.o
SKIPK 		= _release/skipValue_kernal.o
ADDK 		= _release/addEdges_kernal.o
PZER 		= _release/parallel_PZER.o
EXCEP 		= _release/Exceptions.o
GSTM 		= _release/gstream.o
PATHC 		= _release/Path.o
GRAPH  		= _release/Graph.o
GRAPHDRAW  	= _release/GraphDraw.o
EDITOR 		= _release/Editor.o
APP 		= output/bin/main/cuGraph_1.0.0
FUNC_TEST 	= output/bin/test/functional_test

# Shell Commands:
cuda: $(SCANK) $(RANDK) $(SKIPK) $(ADDK) $(PZER) 
all: $(SCANK) $(RANDK) $(SKIPK) $(ADDK) $(PZER) $(EXCEP) $(GSTM) $(PATHC) $(GRAPH) $(GRAPHDRAW) $(EDITOR) $(APP)
functional_test: $(SCANK) $(RANDK) $(SKIPK) $(ADDK) $(PZER) $(EXCEP) $(GSTM) $(PATHC) $(GRAPH) $(GRAPHDRAW) $(EDITOR) $(FUNC_TEST)

run: 
	./$(APP)
	
run_functional_test: 
	./$(FUNC_TEST)
	
clean:
	$(RM) $(shell pwd)/*~ 
	$(RM) $(shell pwd)/*.png 
	$(RM) $(shell pwd)/src/*~ 
	$(RM) $(shell pwd)/src/cuda/*~ 
	$(RM) $(shell pwd)/src/main/*~ 
	$(RM) $(shell pwd)/src/test/*~ 
	$(RM) $(shell pwd)/include/*~ 
	$(RM) $(shell pwd)/include/cuda/*~ 
	$(RM) $(shell pwd)/include/main/*~ 
	$(RM) $(shell pwd)/include/test/*~ 
	$(RM) $(shell pwd)/output/GML/*~
	$(RM) $(shell pwd)/output/TXT/*~
	$(RM) $(shell pwd)/output/MTX/*~
	$(RM) $(shell pwd)/output/bin/main/*
	$(RM) $(shell pwd)/output/bin/test/*
	$(RM) $(shell pwd)/_debug/*
	$(RM) $(shell pwd)/_release/*
		
# Build Commands:	
$(SCANK): src/cuda/scan_kernel.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(RANDK): src/cuda/random_number_generator_kernal.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(SKIPK): src/cuda/skipValue_kernal.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^

$(ADDK): src/cuda/addEdges_kernal.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^

$(PZER): src/cuda/parallel_PZER.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
		
$(EXCEP): src/main/Exceptions.cpp 
	$(CPP) $(INCFLAGS) -o $@ -c $^

$(GSTM): src/main/gstream.cpp
	$(CPP) $(INCFLAGS) -o $@ -c $^
	
$(PATHC): src/main/Path.cpp
	$(CPP) $(INCFLAGS) -o $@ -c $^

$(GRAPH): src/main/Graph.cpp
	$(CPP) $(INCFLAGS) -o $@ -c $^

$(GRAPHDRAW): src/main/GraphDraw.cpp
	$(CPP) $(INCFLAGS) $(INCLQT) -g -Wall $(QTLIBS) -o $@ -c $^
		
$(EDITOR): src/main/Editor.cpp include/main/Editor.h
	$(QT4C) include/main/Editor.h | $(CPP) $(OFLAGS) $(INCFLAGS) -o $@ -c -x c++ - -include src/main/Editor.cpp

$(APP): main.cpp $(SCANK) $(RANDK) $(SKIPK) $(ADDK) $(PZER) $(GRAPH) $(GRAPHDRAW) $(EDITOR) $(PATHC) $(EXCEP) $(GSTM)
	$(CPP) $(CUDALIBS) $(INCFLAGS) $(OFLAGS) $^ $(CPPFLAGS) $(QTLIBS) -o $@
	
$(FUNC_TEST): src/test/maintest.cpp src/test/functional_test.cpp\
$(SCANK) $(RANDK) $(SKIPK) $(ADDK) $(PZER) $(GRAPH) $(GRAPHDRAW) $(EDITOR) $(PATHC) $(EXCEP) $(GSTM)
	$(CPP) $(CUDALIBS) $(INCFLAGS) $(OFLAGS) $^ $(CPPFLAGS) $(QTLIBS) -o $@
	
