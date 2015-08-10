# Compilers Definitions:
CPP  		= g++
NVCC 		= nvcc

# Compilers Flags:
CPPSTAND	= -std=c++11
CPPFLAGS 	= -g -Wall -fopenmp
CUDAFLAGS 	= -arch compute_20 -code sm_20
CUDALIBS 	= -L/usr/local/cuda/lib64 -lcuda -lcudart
INCFLAGS 	= -I./include

# Libraries:
INCLQT 		= -I/usr/include/qt4 -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui
OFLAGS 		= $(INCLQT) -Wall -Wno-unreachable-code -Wno-return-type

# Directories:
PARSE       = _obj/parse.o
UTIL		= _obj/util.o
SCANK		= _obj/scan_kernels.o
COMPK		= _obj/Stream_compaction_kernels.o
RANDK 		= _obj/random_number_generator_kernels.o
SKIPK 		= _obj/skipValue_kernels.o
ADDK 		= _obj/addEdges_kernels.o
PER 		= _obj/PER_Generator.o
PZER 		= _obj/PZER_Generator.o
PPreZER 	= _obj/PPreZER_Generator.o
EXCEP 		= _obj/Exceptions.o
GSTM 		= _obj/gstream.o
PATHC 		= _obj/Path.o
GRAPH  		= _obj/Graph.o
APP 		= output/cuGraph

# Shell Commands:
all: $(PARSE) $(UTIL) $(SCANK) $(COMPK) $(RANDK) $(SKIPK) $(ADDK) $(PER) $(PZER) $(PPreZER) $(EXCEP) $(GSTM) $(PATHC) $(GRAPH) $(APP)

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
	$(RM) $(shell pwd)/output/cuGraph
	$(RM) $(shell pwd)/_obj/*~
		
# Build Commands:	
$(PARSE): src/main/argvparser.cpp
	$(CPP) $(CPPSTAND) $(INCFLAGS) -o $@ -c $^
	
$(UTIL): src/cuda/util.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(SCANK): src/cuda/scan_kernels.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(COMPK): src/cuda/Stream_compaction_kernels.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(RANDK): src/cuda/random_number_generator_kernels.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(SKIPK): src/cuda/skipValue_kernels.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^

$(ADDK): src/cuda/addEdges_kernels.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^

$(PER): src/cuda/PER_Generator.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(PZER): src/cuda/PZER_Generator.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^

$(PPreZER): src/cuda/PPreZER_Generator.cu
	$(NVCC) $(CUDAFLAGS) $(INCFLAGS) -o $@ -c $^
	
$(EXCEP): src/main/Exceptions.cpp 
	$(CPP) $(INCFLAGS) -o $@ -c $^

$(GSTM): src/main/gstream.cpp
	$(CPP) $(INCFLAGS) -o $@ -c $^
	
$(PATHC): src/main/Path.cpp
	$(CPP) $(INCFLAGS) -o $@ -c $^

$(GRAPH): src/main/Graph.cpp
	$(CPP) $(INCFLAGS) -o $@ -c $^

$(APP): main.cpp $(PARSE) $(UTIL) $(SCANK) $(COMPK) $(RANDK) $(SKIPK) $(ADDK) $(PER) $(PZER) $(PPreZER) \
$(GRAPH) $(PATHC) $(EXCEP) $(GSTM)
	$(CPP) $(CUDALIBS) $(INCFLAGS) $(OFLAGS) $^ $(CPPFLAGS) $(CPPSTAND) -o $@








