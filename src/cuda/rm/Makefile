########## the compilers and their flags #################

# the compilers:
CUDACC := nvcc

# flags:
CUDACC_FLAGS :=

# project files:
FILES := 

##################### Build Rules ########################

all: scan.o scan_test

clean: 
	rm -f cuMain

scan.o: scan.cu
	$(CUDACC) -o $@ -c scan.cu -arch compute_20 -code sm_20

scan_test: scan_test.cu scan.o
	$(CUDACC) $^ -o $@  -arch compute_20 -code sm_20
	
##########################################################

