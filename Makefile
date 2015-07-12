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
RANDK 		= _obj/random_number_generator_kernal.o
SKIPK 		= _obj/skipValue_kernal.o
ADDK 		= _obj/addEdges_kernal.o
PZER 		= _obj/parallel_PZER.o
EXCEP 		= _obj/Exceptions.o
GSTM 		= _obj/gstream.o
PATHC 		= _obj/Path.o
GRAPH  		= _obj/Graph.o
APP 		= output/bin/cuGraph_1.0.0

# Shell Commands:
cuda: $(RANDK) $(SKIPK) $(ADDK) $(PZER) 
all: $(RANDK) $(SKIPK) $(ADDK) $(PZER) $(EXCEP) $(GSTM) $(PATHC) $(GRAPH) $(APP)

run: 
	./$(APP)
	
run_functional_test: 
	./$(FUNC_TEST)
	
clean:
	$(RM) $(shell pwd)/*~ 
	$(RM) $(shell pwd)/src/*~ 
	$(RM) $(shell pwd)/src/cuda/*~ 
	$(RM) $(shell pwd)/src/cuda/scan/*~ 
	$(RM) $(shell pwd)/src/main/*~ 
	$(RM) $(shell pwd)/src/test/*~ 
	$(RM) $(shell pwd)/include/*~ 
	$(RM) $(shell pwd)/include/cuda/*~ 
	$(RM) $(shell pwd)/include/main/*~ 
	$(RM) $(shell pwd)/include/test/*~ 
	$(RM) $(shell pwd)/output/GML/*~
	$(RM) $(shell pwd)/output/TXT/*~
	$(RM) $(shell pwd)/output/MTX/*~
	$(RM) $(shell pwd)/output/bin/cuGraph_1.0.0
	$(RM) $(shell pwd)/_obj/*
		
# Build Commands:	
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

$(APP): main.cpp $(RANDK) $(SKIPK) $(ADDK) $(PZER) $(GRAPH) $(PATHC) $(EXCEP) $(GSTM)
	$(CPP) $(CUDALIBS) $(INCFLAGS) $(OFLAGS) $^ $(CPPFLAGS) $(QTLIBS) -o $@
